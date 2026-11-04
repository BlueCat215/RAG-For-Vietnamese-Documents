"""
Module truy xuất (Retrieval) cho tài liệu tiếng Việt.
Hỗ trợ:
1. BM25 Baseline: Sử dụng rank_bm25 với bộ tách từ tiếng Việt (âm tiết hoặc underthesea).
2. Dense Retrieval: Sử dụng mô hình Sentence-Transformers + FAISS Index.
3. Hybrid Search: Hợp nhất kết quả bằng Reciprocal Rank Fusion (RRF).
"""

import os
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import numpy as np


# ==============================================================================
# Tokenizer cho BM25 tiếng Việt
# ==============================================================================
def tokenize_vietnamese(text: str, mode: str = "syllable") -> List[str]:
    """
    Tokenize cho BM25:
    - 'syllable': Tách theo âm tiết (lowercase + split trắng). Nhanh, chuẩn cho baseline.
    - 'word': Dùng underthesea tách từ ghép ("học sinh" -> "học_sinh").
    """
    text = text.lower().strip()
    if mode == "word":
        try:
            from underthesea import word_tokenize
            # Nối từ ghép bằng dấu gạch dưới
            return word_tokenize(text, format="text").split()
        except ImportError:
            pass
    # Mặc định: tách theo âm tiết và lọc ký tự đặc biệt
    words = [w.strip(".,;:?!\"'()[]{}«»") for w in text.split()]
    return [w for w in words if w]


# ==============================================================================
# Lớp BM25 Retriever
# ==============================================================================
class BM25Retriever:
    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75, mode: str = "syllable"):
        self.chunks = chunks
        self.mode = mode
        self.corpus_tokens = [
            tokenize_vietnamese(c["text"], mode=self.mode)
            for c in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens, k1=k1, b=b)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Truy xuất top-k chunk bằng BM25."""
        query_tokens = tokenize_vietnamese(query, mode=self.mode)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            chunk["rank"] = rank + 1
            chunk["retriever"] = "bm25"
            results.append(chunk)

        return results


# ==============================================================================
# Hợp nhất xếp hạng RRF (Reciprocal Rank Fusion)
# ==============================================================================
def reciprocal_rank_fusion(
    results_list: List[List[Dict[str, Any]]], 
    k: int = 60, 
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Hợp nhất nhiều danh sách kết quả (BM25 + Dense) bằng thuật toán RRF:
    RRF_score(d) = sum(1 / (k + rank_r(d)))
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    for results in results_list:
        for rank, item in enumerate(results):
            c_id = item["chunk_id"]
            if c_id not in chunk_map:
                chunk_map[c_id] = item

            # rank trong Python bắt đầu từ 0 -> thứ hạng thật là rank + 1
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (k + (rank + 1)))

    # Sắp xếp theo điểm RRF giảm dần
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    final_results = []
    for rank, (c_id, score) in enumerate(sorted_items):
        res = dict(chunk_map[c_id])
        res["score"] = score
        res["rank"] = rank + 1
        res["retriever"] = "hybrid_rrf"
        final_results.append(res)

    return final_results
