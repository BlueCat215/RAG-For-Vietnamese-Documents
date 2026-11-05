"""
Module chunking cho tài liệu pháp quy tiếng Việt (NLP-21).

Cung cấp 3 chiến lược:
  S1: chunk_fixed_size - Cắt cố định theo số từ (baseline)
  S2: chunk_recursive  - Cắt đệ quy theo dấu phân cách (\n\n -> \n -> câu -> từ)
  S3: chunk_structural - Cắt theo Chương / Điều / Khoản (chiến lược chính)
"""

import os
import re
import json
import bisect
from typing import List, Dict, Any, Tuple, Optional

from src.chunking_utils import (
    normalize_legal_text,
    estimate_token_count,
    get_page_number,
    trim_span,
    make_chunk,
    group_by_doc,
    build_doc_text,
)


# ==============================================================================
# Chiến lược 1 (S1): Cố định theo số từ
# ==============================================================================
def chunk_fixed_size(pages_data: List[Dict[str, Any]], chunk_size: int = 300, overlap: int = 50) -> List[Dict[str, Any]]:
    """Cắt cố định theo số từ trên chuỗi liên tục của tài liệu."""
    all_chunks: List[Dict[str, Any]] = []
    for doc_id, doc_pages in group_by_doc(pages_data).items():
        full_text, page_starts, pages = build_doc_text(doc_pages)
        spans = [m.span() for m in re.finditer(r"\S+", full_text)]
        n = len(spans)
        if n == 0:
            continue

        step = max(1, chunk_size - overlap)
        idx = 0
        for i in range(0, n, step):
            j = min(i + chunk_size, n)
            chunk_text = " ".join(full_text[a:b] for a, b in spans[i:j])
            p_start = get_page_number(spans[i][0], page_starts, pages)
            p_end = get_page_number(spans[j - 1][1] - 1, page_starts, pages)
            heading = f"Trang {p_start}" if p_start == p_end else f"Trang {p_start} - {p_end}"
            all_chunks.append(make_chunk(doc_id, "s1", idx, chunk_text, p_start, p_end, heading, None, "fixed"))
            idx += 1
            if j >= n:
                break
    return all_chunks


# ==============================================================================
# Chiến lược 2 (S2): Cắt đệ quy
# ==============================================================================
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def recursive_split_spans(src: str, lo: int = 0, hi: Optional[int] = None, max_chars: int = 1200,
                          overlap_chars: int = 150, seps: Optional[List[str]] = None) -> List[Tuple[int, int]]:
    """Cắt src[lo:hi] đệ quy theo seps, trả về danh sách offset tuyệt đối (start, end)."""
    if hi is None:
        hi = len(src)
    if seps is None:
        seps = _SEPARATORS
    overlap_chars = max(0, min(overlap_chars, max_chars // 2))

    if hi - lo <= max_chars:
        return [(lo, hi)] if src[lo:hi].strip() else []

    if not seps:
        step = max(1, max_chars - overlap_chars)
        out, s = [], lo
        while s < hi:
            e = min(s + max_chars, hi)
            out.append((s, e))
            if e >= hi:
                break
            s += step
        return out

    sep = seps[0]
    pieces: List[Tuple[int, int]] = []
    pos = lo
    while pos < hi:
        idx = src.find(sep, pos, hi)
        if idx == -1:
            pieces.append((pos, hi))
            break
        pieces.append((pos, idx + len(sep)))
        pos = idx + len(sep)

    spans: List[Tuple[int, int]] = []

    def flush(group: List[Tuple[int, int]]) -> None:
        if not group:
            return
        s, e = group[0][0], group[-1][1]
        if e - s > max_chars:
            spans.extend(recursive_split_spans(src, s, e, max_chars, overlap_chars, seps[1:]))
        elif src[s:e].strip():
            spans.append((s, e))

    cur: List[Tuple[int, int]] = []
    cur_len = 0
    for ps, pe in pieces:
        plen = pe - ps
        if cur and cur_len + plen > max_chars:
            flush(cur)
            carry: List[Tuple[int, int]] = []
            carry_len = 0
            for piece in reversed(cur):
                ln = piece[1] - piece[0]
                if carry_len + ln <= overlap_chars:
                    carry.insert(0, piece)
                    carry_len += ln
                else:
                    break
            if carry_len + plen > max_chars:
                carry, carry_len = [], 0
            cur, cur_len = carry, carry_len
        cur.append((ps, pe))
        cur_len += plen
    flush(cur)
    return spans


def chunk_recursive(pages_data: List[Dict[str, Any]], max_chars: int = 1200, overlap_chars: int = 150) -> List[Dict[str, Any]]:
    """Cắt tài liệu đệ quy theo đoạn/câu."""
    all_chunks: List[Dict[str, Any]] = []
    for doc_id, doc_pages in group_by_doc(pages_data).items():
        full_text, page_starts, pages = build_doc_text(doc_pages)
        idx = 0
        for s, e in recursive_split_spans(full_text, 0, len(full_text), max_chars, overlap_chars):
            s, e = trim_span(full_text, s, e)
            if s >= e:
                continue
            p_start = get_page_number(s, page_starts, pages)
            p_end = get_page_number(e - 1, page_starts, pages)
            heading = f"Trang {p_start}" if p_start == p_end else f"Trang {p_start} - {p_end}"
            all_chunks.append(make_chunk(doc_id, "s2", idx, full_text[s:e], p_start, p_end, heading, None, "recursive"))
            idx += 1
    return all_chunks


# ==============================================================================
# Chiến lược 3 (S3): Cắt theo Chương / Điều / Khoản (Structural)
# ==============================================================================
_ARTICLE_RE = re.compile(r"^[ \t]*(Điều[ \t]+\d+[a-zđ]*\.[ \t]*[^\n]*)", re.MULTILINE)
_CHAPTER_RE = re.compile(
    r"^[ \t]*(Chương[ \t]+(?:[IVXLCDM]+|\d+)\b[^\n]{0,100})(?:\n[ \t]*([^\n\.\:0-9]{3,100}))?[ \t]*$",
    re.MULTILINE,
)
_CLAUSE_RE = re.compile(r"^[ \t]*(\d{1,2})\.[ \t]+", re.MULTILINE)
_RECIPIENT_RE = re.compile(r"^[ \t]*Nơi nhận[ \t]*:", re.MULTILINE)


def _find_chapters(full_text: str) -> List[Tuple[int, str]]:
    chapters = []
    for m in _CHAPTER_RE.finditer(full_text):
        first = m.group(1).strip()
        rest = re.sub(r"^Chương\s+\S+", "", first).strip(" :-.–")
        if rest.endswith((".", ",", ";")) or " này" in rest or " quy định tại " in rest:
            continue
        label = first
        if m.group(2):
            label += " - " + m.group(2).strip()
        chapters.append((m.start(1), label))
    return chapters


def _find_articles(full_text: str) -> List["re.Match[str]"]:
    raw = list(_ARTICLE_RE.finditer(full_text))
    kept = []
    for i, m in enumerate(raw):
        if i + 1 >= len(raw):
            kept.append(m)
            continue
        if len(full_text[m.end(1):raw[i + 1].start(1)].strip()) >= 15:
            kept.append(m)
    return kept


def _find_clauses(full_text: str, lo: int, hi: int) -> List["re.Match[str]"]:
    out, expected = [], 1
    for m in _CLAUSE_RE.finditer(full_text, lo, hi):
        if int(m.group(1)) == expected:
            out.append(m)
            expected += 1
    return out


def _chunk_structural_doc(doc_id: str, doc_pages: List[Dict[str, Any]], max_tokens: int,
                          split_chars: int, split_overlap: int) -> List[Dict[str, Any]]:
    full_text, page_starts, pages = build_doc_text(doc_pages)
    chapters = _find_chapters(full_text)
    chapter_starts = [c[0] for c in chapters]
    articles = _find_articles(full_text)

    if not articles:
        fb = chunk_recursive(doc_pages)
        for c in fb:
            c["strategy"] = "structural_fallback_recursive"
        return fb

    chunks: List[Dict[str, Any]] = []
    counter = 0

    def span_pages(s: int, e: int) -> Tuple[int, int]:
        return (get_page_number(s, page_starts, pages),
                get_page_number(max(s, e - 1), page_starts, pages))

    def emit(text: str, p_start: int, p_end: int, heading: str, clause: Optional[str], strategy: str) -> None:
        nonlocal counter
        chunks.append(make_chunk(doc_id, "s3", counter, text, p_start, p_end, heading, clause, strategy))
        counter += 1

    def emit_pieces(prefix: str, lo: int, hi: int, heading: str, clause: Optional[str], strategy: str) -> None:
        budget = max(300, split_chars - len(prefix))
        for s, e in recursive_split_spans(full_text, lo, hi, budget, split_overlap):
            s, e = trim_span(full_text, s, e)
            if s >= e:
                continue
            ps, pe = span_pages(s, e)
            body = full_text[s:e]
            emit(f"{prefix}\n{body}" if prefix else body, ps, pe, heading, clause, strategy)

    # 1. Phần mở đầu trước Điều 1
    pre_s, pre_e = trim_span(full_text, 0, articles[0].start(1))
    if pre_e - pre_s >= 40:
        pre_text = full_text[pre_s:pre_e]
        heading = "Căn cứ & Quy định chung"
        if estimate_token_count(pre_text) <= max_tokens:
            ps, pe = span_pages(pre_s, pre_e)
            emit(pre_text, ps, pe, heading, None, "structural_preamble")
        else:
            emit_pieces("", pre_s, pre_e, heading, None, "structural_preamble")

    # 2. Duyệt qua từng Điều
    for i, m in enumerate(articles):
        art_start = m.start(1)
        title_end = m.end(1)
        next_art_start = articles[i + 1].start(1) if i + 1 < len(articles) else len(full_text)
        art_end = next_art_start

        ch_idx = bisect.bisect_right(chapter_starts, art_start)
        if ch_idx < len(chapter_starts) and chapter_starts[ch_idx] < next_art_start:
            art_end = chapter_starts[ch_idx]
        if i + 1 >= len(articles):
            rm = _RECIPIENT_RE.search(full_text, title_end)
            if rm and rm.start() < art_end:
                art_end = rm.start()

        article_title = m.group(1).strip()
        current_chapter = chapters[ch_idx - 1][1] if ch_idx - 1 >= 0 else "Quy định chung"
        parent_header = f"{current_chapter} - {article_title}"

        art_s, art_e = trim_span(full_text, art_start, art_end)
        article_body = full_text[art_s:art_e]
        p_start, p_end = span_pages(art_s, art_e)

        # 2a. Điều vừa vặn
        whole_text = f"{current_chapter}\n{article_body}"
        if estimate_token_count(whole_text) <= max_tokens:
            emit(whole_text, p_start, p_end, parent_header, None, "structural")
            continue

        # 2b. Điều dài -> tách theo Khoản
        clause_matches = _find_clauses(full_text, title_end, art_end)
        if clause_matches:
            for c_i, cm in enumerate(clause_matches):
                c_start = title_end if c_i == 0 else cm.start(1)
                c_end = clause_matches[c_i + 1].start(1) if c_i + 1 < len(clause_matches) else art_end
                s, e = trim_span(full_text, c_start, c_end)
                if s >= e:
                    continue
                clause_label = f"Khoản {cm.group(1)}"
                prefix = f"{parent_header} > {clause_label}"
                candidate = f"{prefix}\n{full_text[s:e]}"
                if estimate_token_count(candidate) <= max_tokens:
                    ps, pe = span_pages(s, e)
                    emit(candidate, ps, pe, parent_header, clause_label, "structural_clause")
                else:
                    emit_pieces(prefix, s, e, parent_header, clause_label, "structural_clause_split")
        else:
            emit_pieces(parent_header, title_end, art_end, parent_header, None, "structural_split")

    return chunks


def chunk_structural(pages_data: List[Dict[str, Any]], max_tokens: int = 512,
                     split_chars: int = 1200, split_overlap: int = 100) -> List[Dict[str, Any]]:
    """Chiến lược chunking cấu trúc chính (Chương / Điều / Khoản)."""
    all_chunks: List[Dict[str, Any]] = []
    for doc_id, doc_pages in group_by_doc(pages_data).items():
        all_chunks.extend(_chunk_structural_doc(doc_id, doc_pages, max_tokens, split_chars, split_overlap))
    return all_chunks


def save_chunks_to_jsonl(chunks: List[Dict[str, Any]], output_path: str) -> None:
    """Lưu chunks ra JSONL an toàn."""
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
