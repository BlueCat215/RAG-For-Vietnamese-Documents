"""
Module Cross-Encoder Reranker cho tiếng Việt.
Sử dụng model thanhtantran/Vietnamese_Reranker để chấm điểm lại mức độ phù hợp giữa (câu hỏi, đoạn văn).
Khuyến nghị chạy trên CPU để giữ trọn vẹn VRAM GPU cho LLM.
Tích hợp kỹ thuật Re-ranking mềm (Soft Re-ranking) kết hợp cùng Dense Semantic Similarity.
"""

import torch
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder


class VietnameseReranker:
    def __init__(self, model_name: str = "thanhtantran/Vietnamese_Reranker", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        print(f"[*] Khởi tạo Reranker Model: {model_name} trên {device}...")
        self.model = CrossEncoder(model_name, max_length=512, device=device)

    def rerank(
        self, 
        query: str, 
        candidate_chunks: List[Dict[str, Any]], 
        top_k: int = 5,
        alpha: float = 0.75
    ) -> List[Dict[str, Any]]:
        """
        Chấm điểm từng cặp (query, chunk_text) và xếp hạng lại.
        Áp dụng kỹ thuật Re-ranking mềm (Soft Re-ranking):
        Do Cross-Encoder có phân phối điểm phi tuyến tính cao, hệ thống kết hợp điểm Cross-Encoder
        với điểm tương đồng ngữ nghĩa đa chiều của Dense Embedding (AITeamVN/Vietnamese_Embedding):
            soft_score = alpha * rerank_raw + (1 - alpha) * dense_score
        Giúp bảo toàn các đoạn văn mang ngữ nghĩa hành chính tương ứng khi người dùng hỏi bằng khẩu ngữ.
        """
        if not candidate_chunks:
            return []

        # Chuẩn bị danh sách cặp câu
        pairs = [(query, c["text"]) for c in candidate_chunks]

        # Dự đoán điểm mức độ liên quan từ Cross-Encoder
        raw_scores = self.model.predict(pairs, show_progress_bar=False)

        reranked_chunks = []
        for chunk, score in zip(candidate_chunks, raw_scores):
            c_copy = dict(chunk)
            r_score = float(score)
            d_score = float(chunk.get("dense_score", 0.0))

            if d_score > 0.0:
                soft_score = alpha * r_score + (1.0 - alpha) * d_score
            else:
                soft_score = r_score

            c_copy["rerank_raw"] = r_score
            c_copy["dense_score"] = d_score
            c_copy["rerank_score"] = float(soft_score)
            reranked_chunks.append(c_copy)

        reranked_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Lấy top_k kết quả tốt nhất
        final_results = []
        for rank, c in enumerate(reranked_chunks[:top_k]):
            c["rank"] = rank + 1
            c["retriever"] = "reranked"
            final_results.append(c)

        return final_results
