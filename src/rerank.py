"""
Module Cross-Encoder Reranker cho tiếng Việt.
Sử dụng model reranker để chấm điểm lại mức độ phù hợp giữa (câu hỏi, đoạn văn).
Khuyến nghị chạy trên CPU để giữ trọn vẹn VRAM GPU cho LLM.
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

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Chấm điểm từng cặp (query, chunk_text) và sắp xếp lại.
        """
        if not candidate_chunks:
            return []

        # Chuẩn bị danh sách cặp câu
        pairs = [(query, c["text"]) for c in candidate_chunks]

        # Dự đoán điểm mức độ liên quan
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Gán lại điểm và sắp xếp theo score giảm dần
        reranked_chunks = []
        for chunk, score in zip(candidate_chunks, scores):
            c_copy = dict(chunk)
            c_copy["rerank_score"] = float(score)
            reranked_chunks.append(c_copy)

        reranked_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Lấy top_k kết quả tốt nhất
        final_results = []
        for rank, c in enumerate(reranked_chunks[:top_k]):
            c["rank"] = rank + 1
            c["retriever"] = "reranked"
            final_results.append(c)

        return final_results
