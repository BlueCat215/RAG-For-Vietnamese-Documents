"""
Pipeline RAG End-to-End cho tài liệu tiếng Việt (NLP-21).

Tích hợp toàn bộ luồng:
Query -> (BM25 / Dense / Hybrid RRF) -> Reranking -> Refusal Check (Tau) -> LLM Generation -> Output + Nguồn

Xử lý ngoại lệ và tối ưu:
1. Cơ chế từ chối 2 lớp (Two-tier Refusal):
   - Lớp 1 (Truy hồi): Điểm top-1 thấp hơn ngưỡng tau -> Dừng ngay, trả lời "Không tìm thấy"
     (KHÔNG gọi LLM -> Tiết kiệm 100% thời gian sinh và VRAM).
   - Lớp 2 (Sinh câu trả lời): LLM tự nhận thức context không đủ -> Từ chối qua prompt.
2. Quản lý tài nguyên cho RTX 3050 4GB:
   - Embedding query chạy trên CPU (nhanh, 0MB VRAM).
   - Reranker chạy trên CPU (nhẹ, 0MB VRAM).
   - LLM chiếm dụng GPU.
3. Đo đạc độ trễ (latency breakdown):
   - Tách riêng thời gian: retrieve_ms, rerank_ms, generate_ms.
"""

import time
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

from src.utils import load_config
from src.retrieve import BM25Retriever, reciprocal_rank_fusion
from src.index import VectorIndex
from src.rerank import VietnameseReranker
from src.generate import VietnameseLLMGenerator


class RAGPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.paths = self.config["paths"]
        self.models_cfg = self.config["models"]
        self.retrieval_cfg = self.config["retrieval"]

        # 1. Khởi tạo FAISS Index
        emb_cfg = self.models_cfg["embedding"]
        self.vector_index = VectorIndex(model_name=emb_cfg["name"], device=emb_cfg.get("device", "cuda"))
        
        # 2. Khởi tạo embedding model cho truy vấn (chạy trên CPU để tiết kiệm VRAM cho LLM)
        self.query_encoder = None
        self.emb_model_name = emb_cfg["name"]

        # 3. Khởi tạo BM25 (sẽ nạp sau khi index được load)
        self.bm25_retriever = None

        # 4. Khởi tạo Reranker (chạy trên CPU)
        rerank_cfg = self.models_cfg["reranker"]
        self.reranker = VietnameseReranker(
            model_name=rerank_cfg["name"],
            device=rerank_cfg.get("device", "cpu")
        )
        self.top_n_rerank = rerank_cfg.get("top_n_to_rerank", 20)
        self.top_k_final = rerank_cfg.get("top_k_final", 5)

        # 5. Khởi tạo LLM Generator (chạy GPU 4-bit)
        llm_cfg = self.models_cfg["llm"]
        self.generator = VietnameseLLMGenerator(
            model_name=llm_cfg["name"],
            quantization=llm_cfg.get("quantization", "4bit"),
            device=llm_cfg.get("device", "cuda"),
            max_new_tokens=llm_cfg.get("max_new_tokens", 350),
            temperature=llm_cfg.get("temperature", 0.1)
        )

        # Ngưỡng từ chối tau
        self.refusal_threshold = float(self.retrieval_cfg.get("refusal_threshold", 0.45))
        self.is_ready = False

    def initialize(self, index_dir: Optional[str] = None):
        """Nạp chỉ mục và chuẩn bị sẵn sàng cho truy vấn."""
        idx_dir = index_dir or self.paths["index_dir"]
        print(f"[*] Đang nạp chỉ mục từ: {idx_dir}...")
        self.vector_index.load(idx_dir)

        # Khởi tạo mô hình encode câu hỏi trên CPU
        print(f"[*] Nạp Query Encoder ({self.emb_model_name}) trên CPU...")
        self.query_encoder = SentenceTransformer(self.emb_model_name, device="cpu")

        # Khởi tạo BM25 trên tập chunks đã nạp
        print(f"[*] Khởi tạo BM25 trên {len(self.vector_index.chunks)} chunks...")
        bm25_cfg = self.retrieval_cfg["bm25"]
        self.bm25_retriever = BM25Retriever(
            self.vector_index.chunks,
            k1=bm25_cfg.get("k1", 1.5),
            b=bm25_cfg.get("b", 0.75),
            mode="syllable"
        )

        self.is_ready = True
        print("[+] Pipeline RAG đã sẵn sàng phục vụ!")

    def retrieve(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 5,
        use_rerank: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Truy xuất các đoạn văn liên quan nhất:
        - method: 'bm25' | 'dense' | 'hybrid'
        - use_rerank: True | False (dùng để chạy thí nghiệm ablation)
        """
        if not self.is_ready:
            raise RuntimeError("Pipeline chưa được initialize()!")

        timings = {}
        t0 = time.time()

        # 1. Tìm kiếm theo phương pháp
        if method == "bm25":
            candidates = self.bm25_retriever.search(query, top_k=self.top_n_rerank if use_rerank else top_k)
        elif method == "dense":
            candidates = self.vector_index.search(query, self.query_encoder, top_k=self.top_n_rerank if use_rerank else top_k)
        elif method == "hybrid":
            bm25_results = self.bm25_retriever.search(query, top_k=self.top_n_rerank)
            dense_results = self.vector_index.search(query, self.query_encoder, top_k=self.top_n_rerank)
            candidates = reciprocal_rank_fusion(
                [bm25_results, dense_results],
                k=self.retrieval_cfg.get("rrf", {}).get("k", 60),
                top_k=self.top_n_rerank if use_rerank else top_k
            )
        else:
            raise ValueError(f"Phương thức không hợp lệ: {method}")

        timings["retrieve_ms"] = (time.time() - t0) * 1000

        # 2. Xếp hạng lại (Reranking)
        if use_rerank and candidates:
            t1 = time.time()
            final_chunks = self.reranker.rerank(query, candidates, top_k=top_k)
            timings["rerank_ms"] = (time.time() - t1) * 1000
        else:
            final_chunks = candidates[:top_k]
            timings["rerank_ms"] = 0.0

        return final_chunks, timings

    def query(
        self,
        question: str,
        method: str = "hybrid",
        top_k: int = 5,
        use_rerank: bool = True
    ) -> Dict[str, Any]:
        """
        Hàm xử lý truy vấn hỏi đáp hoàn chỉnh từ đầu đến cuối (End-to-End).
        """
        question = question.strip()
        if not question:
            return {
                "question": question,
                "answer": "Vui lòng nhập câu hỏi hợp lệ.",
                "refused": True,
                "sources": [],
                "latency_ms": {}
            }

        # Bước 1: Truy xuất và xếp hạng
        contexts, timings = self.retrieve(question, method=method, top_k=top_k, use_rerank=use_rerank)

        # Bước 2: Kiểm tra ngưỡng từ chối Lớp 1 (Refusal Gate)
        # Nếu điểm cao nhất thấp hơn ngưỡng tau -> Không cần gọi LLM
        max_score = 0.0
        if contexts:
            # Ưu tiên lấy rerank_score nếu có, ngược lại lấy score của retriever
            max_score = contexts[0].get("rerank_score", contexts[0].get("score", 0.0))

        if not contexts or max_score < self.refusal_threshold:
            return {
                "question": question,
                "answer": "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp.",
                "refused": True,
                "refusal_reason": f"Điểm tương đồng cao nhất ({max_score:.3f}) dưới ngưỡng từ chối ({self.refusal_threshold}).",
                "sources": [],
                "contexts": contexts,
                "latency_ms": {**timings, "generate_ms": 0.0, "total_ms": sum(timings.values())}
            }

        # Bước 3: Gọi LLM sinh câu trả lời
        t2 = time.time()
        gen_res = self.generator.generate(question, contexts)
        timings["generate_ms"] = (time.time() - t2) * 1000
        timings["total_ms"] = sum(timings.values())

        # Bước 4: Ánh xạ mã trích dẫn [n] sang thông tin nguồn chi tiết
        sources = []
        for cite_idx in gen_res["citations"]:
            if 1 <= cite_idx <= len(contexts):
                c = contexts[cite_idx - 1]
                sources.append({
                    "citation_id": cite_idx,
                    "doc_id": c.get("doc_id"),
                    "page_start": c.get("page_start"),
                    "page_end": c.get("page_end"),
                    "heading": c.get("heading"),
                    "clause": c.get("clause"),
                    "score": c.get("rerank_score", c.get("score")),
                    "snippet": c.get("text")[:200] + "..."
                })

        return {
            "question": question,
            "answer": gen_res["answer"],
            "refused": gen_res["refused"],
            "sources": sources,
            "contexts": contexts,
            "citations_valid": gen_res["citations"],
            "citations_hallucinated": gen_res["hallucinated_citations"],
            "latency_ms": timings
        }
