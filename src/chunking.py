"""
Module chunking cho tài liệu pháp quy tiếng Việt (đề tài NLP-21 - RAG).

Ba chiến lược:
  S1 - chunk_fixed_size : cắt cố định theo số từ, có overlap (baseline)
  S2 - chunk_recursive  : cắt đệ quy \n\n -> \n -> ". " -> " " -> ký tự, có overlap
  S3 - chunk_structural : cắt theo Chương / Điều / Khoản (chiến lược chính)

Đầu vào chung: pages_data = [{"doc_id": str, "page": int, "text": str}, ...]
Đầu ra: list chunk dạng dict (xem make_chunk), lưu bằng save_chunks_to_jsonl.
"""
import os
import re
import json
import math
import bisect
import hashlib
import unicodedata
from typing import List, Dict, Any, Tuple, Optional


# ==============================================================================
# Helper functions
# ==============================================================================

def normalize_legal_text(text: Optional[str]) -> str:
    """
    Chuẩn hóa văn bản pháp luật tiếng Việt từ PDF:
    - Unicode NFC.
    - 'Ð' (U+00D0) -> 'Đ', 'ð' -> 'đ' (lỗi font hay gặp).
    - Chuẩn hóa xuống dòng và khoảng trắng không ngắt (\\xa0).
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("Ð", "Đ").replace("ð", "đ")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    return text


def hash_text(text: str) -> str:
    """Mã băm ngắn định danh chunk (không dùng cho bảo mật)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def estimate_token_count(text: str) -> int:
    """
    Ước lượng số token cho tiếng Việt: khoảng 1.3 token trên mỗi âm tiết
    (âm tiết = chuỗi cách nhau bởi khoảng trắng). Nếu có thể, hãy thay bằng
    tokenizer thật của model embedding để khớp giới hạn đầu vào.
    """
    return int(len(text.strip().split()) * 1.3)


def get_page_number(char_pos: int, page_starts: List[int], pages: List[int]) -> int:
    """Tìm số trang theo vị trí ký tự bằng bisect, O(log N)."""
    idx = bisect.bisect_right(page_starts, char_pos) - 1
    idx = max(0, min(idx, len(pages) - 1))
    return pages[idx]


def _trim_span(src: str, s: int, e: int) -> Tuple[int, int]:
    """Bỏ khoảng trắng hai đầu nhưng vẫn giữ offset tuyệt đối trong src."""
    while s < e and src[s].isspace():
        s += 1
    while e > s and src[e - 1].isspace():
        e -= 1
    return s, e


def make_chunk(doc_id: str, tag: str, idx: int, text: str, page_start: int, page_end: int,
               heading: str, clause: Optional[str], strategy: str) -> Dict[str, Any]:
    return {
        "chunk_id": f"{doc_id}_{tag}_{idx:04d}_{hash_text(text)}",
        "doc_id": doc_id,
        "page_start": page_start,
        "page_end": page_end,
        "heading": heading,
        "clause": clause,
        "text": text,
        "strategy": strategy,
        "estimated_tokens": estimate_token_count(text),
    }


def group_by_doc(pages_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    docs: Dict[str, List[Dict[str, Any]]] = {}
    for p in pages_data:
        docs.setdefault(p["doc_id"], []).append(p)
    return docs


_PAGE_NUM_RE = re.compile(r"^(?:Trang\s+)?\d{1,4}(?:\s*/\s*\d{1,4})?$", re.IGNORECASE)
_STRUCT_LINE_RE = re.compile(r"^(?:Điều|Chương|Mục)\b")


def clean_page_texts(doc_pages: List[Dict[str, Any]], min_pages: int = 4,
                     ratio: float = 0.6, max_len: int = 100) -> List[str]:
    """
    Chuẩn hóa và loại bỏ header/footer lặp lại ở >= ratio số trang
    (tên văn bản, số công báo...) cùng các dòng chỉ chứa số trang ("Trang 3", "3/12").
    """
    texts = [normalize_legal_text(p.get("text")) for p in doc_pages]
    repeated = set()
    if len(texts) >= min_pages:
        counts: Dict[str, int] = {}
        for t in texts:
            for ln in {x.strip() for x in t.split("\n") if x.strip()}:
                if len(ln) <= max_len:
                    counts[ln] = counts.get(ln, 0) + 1
        threshold = max(2, math.ceil(len(texts) * ratio))
        repeated = {ln for ln, c in counts.items()
                    if c >= threshold and not _STRUCT_LINE_RE.match(ln)}

    cleaned = []
    for t in texts:
        kept = []
        for ln in t.split("\n"):
            s = ln.strip()
            if s and (s in repeated or _PAGE_NUM_RE.match(s)):
                continue
            kept.append(ln)
        cleaned.append("\n".join(kept))
    return cleaned


def build_doc_text(doc_pages: List[Dict[str, Any]], clean: bool = True) -> Tuple[str, List[int], List[int]]:
    """Ghép các trang của MỘT tài liệu thành một chuỗi, kèm offset bắt đầu của từng trang."""
    doc_pages = sorted(doc_pages, key=lambda x: x["page"])
    if clean:
        texts = clean_page_texts(doc_pages)
    else:
        texts = [normalize_legal_text(p.get("text")) for p in doc_pages]

    parts, page_starts, pages = [], [], []
    pos = 0
    for p, t in zip(doc_pages, texts):
        seg = t + "\n\n"
        page_starts.append(pos)
        pages.append(p["page"])
        parts.append(seg)
        pos += len(seg)
    return "".join(parts), page_starts, pages


# ==============================================================================
# Chiến lược 1 (S1): Cố định theo số từ
# ==============================================================================
def chunk_fixed_size(pages_data: List[Dict[str, Any]], chunk_size: int = 300, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Cắt cố định theo số từ trên chuỗi liên tục của tài liệu (không ngắt cứng theo trang).
    Offset ký tự lấy từ vị trí thật của từng từ nên số trang chính xác.
    """
    all_chunks: List[Dict[str, Any]] = []
    for doc_id, doc_pages in group_by_doc(pages_data).items():
        full_text, page_starts, pages = build_doc_text(doc_pages)
        spans = [m.span() for m in re.finditer(r"\S+", full_text)]
        n = len(spans)
        if n == 0:
            continue

        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size

        idx = 0
        for i in range(0, n, step):
            j = min(i + chunk_size, n)
            chunk_text = " ".join(full_text[a:b] for a, b in spans[i:j])
            p_start = get_page_number(spans[i][0], page_starts, pages)
            p_end = get_page_number(spans[j - 1][1] - 1, page_starts, pages)
            heading = f"Trang {p_start}" if p_start == p_end else f"Trang {p_start} - {p_end}"
            all_chunks.append(make_chunk(doc_id, "s1", idx, chunk_text, p_start, p_end,
                                         heading, None, "fixed"))
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
    """
    Cắt src[lo:hi] theo thứ tự ưu tiên của seps, trả về danh sách (start, end) là offset TUYỆT ĐỐI
    trong src (nên không cần find() để dò lại vị trí). Các mảnh liền kề có overlap tối đa overlap_chars.
    Mảnh nào vẫn dài hơn max_chars sẽ được cắt tiếp bằng separator kế tiếp; cuối cùng cắt cứng theo ký tự.
    """
    if hi is None:
        hi = len(src)
    if seps is None:
        seps = _SEPARATORS
    overlap_chars = max(0, min(overlap_chars, max_chars // 2))

    if hi - lo <= max_chars:
        return [(lo, hi)] if src[lo:hi].strip() else []

    if not seps:  # cắt cứng theo ký tự
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
            if carry_len + plen > max_chars:  # overlap làm tràn thì bỏ overlap
                carry, carry_len = [], 0
            cur, cur_len = carry, carry_len
        cur.append((ps, pe))
        cur_len += plen
    flush(cur)
    return spans


def recursive_split_text(text: str, max_chars: int = 1200, overlap_chars: int = 150) -> List[str]:
    """Phiên bản trả về chuỗi (đã strip) của recursive_split_spans."""
    out = []
    for s, e in recursive_split_spans(text, 0, len(text), max_chars, overlap_chars):
        s, e = _trim_span(text, s, e)
        if s < e:
            out.append(text[s:e])
    return out


def chunk_recursive(pages_data: List[Dict[str, Any]], max_chars: int = 1200, overlap_chars: int = 150) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []
    for doc_id, doc_pages in group_by_doc(pages_data).items():
        full_text, page_starts, pages = build_doc_text(doc_pages)
        idx = 0
        for s, e in recursive_split_spans(full_text, 0, len(full_text), max_chars, overlap_chars):
            s, e = _trim_span(full_text, s, e)
            if s >= e:
                continue
            p_start = get_page_number(s, page_starts, pages)
            p_end = get_page_number(e - 1, page_starts, pages)
            heading = f"Trang {p_start}" if p_start == p_end else f"Trang {p_start} - {p_end}"
            all_chunks.append(make_chunk(doc_id, "s2", idx, full_text[s:e], p_start, p_end,
                                         heading, None, "recursive"))
            idx += 1
    return all_chunks


# ==============================================================================
# Chiến lược 3 (S3): Chương / Điều / Khoản
# ==============================================================================
# Điều: bắt buộc chữ 'Đ' hoa, có dấu chấm sau số, chấp nhận Điều 3a, 3b.
# Không dùng IGNORECASE và không dùng \s để tránh bắt nhầm tham chiếu "điều 10 của..." ở đầu dòng.
_ARTICLE_RE = re.compile(r"^[ \t]*(Điều[ \t]+\d+[a-zđ]*\.[ \t]*[^\n]*)", re.MULTILINE)
# Chương: dòng đầu <= 100 ký tự sau số, kèm dòng tên chương ngay dưới nếu có.
_CHAPTER_RE = re.compile(
    r"^[ \t]*(Chương[ \t]+(?:[IVXLCDM]+|\d+)\b[^\n]{0,100})(?:\n[ \t]*([^\n\.\:0-9]{3,100}))?[ \t]*$",
    re.MULTILINE,
)
_CLAUSE_RE = re.compile(r"^[ \t]*(\d{1,2})\.[ \t]+", re.MULTILINE)
_RECIPIENT_RE = re.compile(r"^[ \t]*Nơi nhận[ \t]*:", re.MULTILINE)


def _find_chapters(full_text: str) -> List[Tuple[int, str]]:
    """Trả về [(vị trí bắt đầu, nhãn Chương)], đã loại các dòng giống câu tham chiếu."""
    chapters = []
    for m in _CHAPTER_RE.finditer(full_text):
        first = m.group(1).strip()
        rest = re.sub(r"^Chương\s+\S+", "", first).strip(" :-.–")
        if rest.endswith((".", ",", ";")) or " này" in rest or " quy định tại " in rest:
            continue  # tham chiếu chéo, không phải tiêu đề Chương
        label = first
        if m.group(2):
            label += " - " + m.group(2).strip()
        chapters.append((m.start(1), label))
    return chapters


def _find_articles(full_text: str) -> List["re.Match[str]"]:
    """
    Tìm các Điều. Loại các dòng của mục lục: Điều không có nội dung nào
    (>= 15 ký tự) giữa tiêu đề và Điều kế tiếp. Điều cuối luôn được giữ.
    """
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
    """Chỉ giữ các Khoản đánh số tăng dần 1, 2, 3... để tránh bắt nhầm danh sách số khác."""
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
        # Văn bản không chia Điều: fallback về S2
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
        """Chia full_text[lo:hi] thành các mảnh nhỏ; MỖI mảnh đều được gắn prefix (ngữ cảnh cha)."""
        budget = max(300, split_chars - len(prefix))
        for s, e in recursive_split_spans(full_text, lo, hi, budget, split_overlap):
            s, e = _trim_span(full_text, s, e)
            if s >= e:
                continue
            ps, pe = span_pages(s, e)
            body = full_text[s:e]
            emit(f"{prefix}\n{body}" if prefix else body, ps, pe, heading, clause, strategy)

    # 1. Phần mở đầu trước Điều đầu tiên (tên văn bản, căn cứ...)
    pre_s, pre_e = _trim_span(full_text, 0, articles[0].start(1))
    if pre_e - pre_s >= 40:
        pre_text = full_text[pre_s:pre_e]
        heading = "Căn cứ & Quy định chung"
        if estimate_token_count(pre_text) <= max_tokens:
            ps, pe = span_pages(pre_s, pre_e)
            emit(pre_text, ps, pe, heading, None, "structural_preamble")
        else:
            emit_pieces("", pre_s, pre_e, heading, None, "structural_preamble")

    # 2. Từng Điều
    for i, m in enumerate(articles):
        art_start = m.start(1)
        title_end = m.end(1)
        next_art_start = articles[i + 1].start(1) if i + 1 < len(articles) else len(full_text)
        art_end = next_art_start

        # Cắt tại đầu Chương mới nếu nó nằm giữa Điều này và Điều kế tiếp
        ch_idx = bisect.bisect_right(chapter_starts, art_start)
        if ch_idx < len(chapter_starts) and chapter_starts[ch_idx] < next_art_start:
            art_end = chapter_starts[ch_idx]
        # Điều cuối: cắt trước khối "Nơi nhận:" / chữ ký
        if i + 1 >= len(articles):
            rm = _RECIPIENT_RE.search(full_text, title_end)
            if rm and rm.start() < art_end:
                art_end = rm.start()

        article_title = m.group(1).strip()
        current_chapter = chapters[ch_idx - 1][1] if ch_idx - 1 >= 0 else "Quy định chung"
        parent_header = f"{current_chapter} - {article_title}"

        art_s, art_e = _trim_span(full_text, art_start, art_end)
        article_body = full_text[art_s:art_e]  # đã chứa dòng tiêu đề Điều
        p_start, p_end = span_pages(art_s, art_e)

        # 2a. Điều vừa vặn: giữ trọn vẹn
        whole_text = f"{current_chapter}\n{article_body}"
        if estimate_token_count(whole_text) <= max_tokens:
            emit(whole_text, p_start, p_end, parent_header, None, "structural")
            continue

        # 2b. Điều dài: tách theo Khoản
        clause_matches = _find_clauses(full_text, title_end, art_end)
        if clause_matches:
            for c_i, cm in enumerate(clause_matches):
                # Khoản 1 lấy luôn đoạn dẫn nằm giữa tiêu đề Điều và Khoản 1
                c_start = title_end if c_i == 0 else cm.start(1)
                c_end = clause_matches[c_i + 1].start(1) if c_i + 1 < len(clause_matches) else art_end
                s, e = _trim_span(full_text, c_start, c_end)
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
            # 2c. Không nhận được Khoản: chia theo kích thước, mỗi mảnh gắn header Chương - Điều
            emit_pieces(parent_header, title_end, art_end, parent_header, None, "structural_split")

    return chunks


def chunk_structural(pages_data: List[Dict[str, Any]], max_tokens: int = 512,
                     split_chars: int = 1200, split_overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Chunk theo cấu trúc pháp quy. Mỗi tài liệu (doc_id) được xử lý riêng.
    - Phần mở đầu trước Điều 1 được giữ lại.
    - Điều <= max_tokens: giữ nguyên 1 chunk (kèm tên Chương).
    - Điều dài: tách theo Khoản (có trường 'clause', trang riêng cho từng Khoản);
      Khoản vẫn dài thì chia nhỏ theo ký tự, mỗi mảnh đều mang header "Chương - Điều > Khoản".
    """
    all_chunks: List[Dict[str, Any]] = []
    for doc_id, doc_pages in group_by_doc(pages_data).items():
        all_chunks.extend(_chunk_structural_doc(doc_id, doc_pages, max_tokens, split_chars, split_overlap))
    return all_chunks


# ==============================================================================
# Lưu trữ
# ==============================================================================
def save_chunks_to_jsonl(chunks: List[Dict[str, Any]], output_path: str) -> None:
    """Lưu chunks ra JSONL; an toàn cả khi output_path không chứa thư mục cha."""
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
