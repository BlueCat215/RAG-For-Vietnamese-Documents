"""
Giao diện Web Hỏi Đáp Quy Chế Đào Tạo Đại Học (RAG Tiếng Việt - NLP-21).
Sử dụng Streamlit.
"""

import os
import streamlit as st
from src.pipeline import RAGPipeline
from src.extract import extract_document
from src.chunking import chunk_structural, save_chunks_to_jsonl
from src.index import VectorIndex
from src.utils import load_config


# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Hỏi Đáp Quy Chế Đào Tạo (RAG Tiếng Việt)",
    page_icon="🎓",
    layout="wide"
)


@st.cache_resource
def get_pipeline():
    """Khởi tạo và cache RAG Pipeline một lần duy nhất."""
    p = RAGPipeline(config_path="config/config.yaml")
    p.initialize()
    return p


# ==============================================================================
# Giao diện chính
# ==============================================================================
st.title("🎓 Trợ Lý Hỏi Đáp Quy Chế Đào Tạo Đại Học")
st.caption("Hệ thống RAG tiếng Việt hỗ trợ sinh viên tra cứu quy chế, học phí, học lại và tốt nghiệp")

# Sidebar: Quản lý và cấu hình
with st.sidebar:
    st.header("⚙️ Cấu hình truy xuất")
    retriever_mode = st.selectbox(
        "Phương thức tìm kiếm:",
        options=["hybrid", "dense", "bm25"],
        format_func=lambda x: {
            "hybrid": "Hybrid Search (BM25 + Dense RRF)",
            "dense": "Dense Vector Search (Ngữ nghĩa)",
            "bm25": "BM25 Baseline (Từ khóa)"
        }[x]
    )
    use_rerank = st.checkbox("Bật Reranker (Cross-Encoder)", value=True)
    top_k = st.slider("Số lượng đoạn trích dẫn (Top-K):", min_value=1, max_value=8, value=4)

    st.divider()
    st.header("📁 Tài liệu trong hệ thống")
    cfg = load_config()
    raw_dir = cfg["paths"]["raw_dir"]
    if os.path.exists(raw_dir):
        files = [f for f in os.listdir(raw_dir) if f.endswith(('.pdf', '.docx', '.txt'))]
        for f in files:
            st.text(f"📄 {f}")
    else:
        st.info("Chưa có tài liệu.")

# Khởi tạo pipeline
try:
    with st.spinner("Đang nạp mô hình và chỉ mục..."):
        pipeline = get_pipeline()
except Exception as e:
    st.error(f"Lỗi khởi tạo Pipeline: {e}. Vui lòng kiểm tra xem bạn đã chạy 'python run_indexing.py' chưa.")
    st.stop()

# Khởi tạo lịch sử chat trong Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Bạn cần tra cứu thông tin gì về quy chế đào tạo, đăng ký học phần, tính điểm rèn luyện hay điều kiện miễn giảm học phí?"}
    ]


def display_sources(sources: list, expanded: bool = True):
    """Hiển thị chi tiết căn cứ và đoạn văn bản nguyên văn RAG đã truy hồi để đối chiếu."""
    if not sources:
        return
    with st.expander("📚 Căn cứ trích dẫn & Đoạn văn bản gốc từ tài liệu (Bằng chứng RAG)", expanded=expanded):
        for s in sources:
            cid = s.get("citation_id", "?")
            doc = s.get("doc_id", "Tài liệu")
            heading = s.get("heading", "")
            clause = s.get("clause", "")
            p_start = s.get("page_start", 1)
            p_end = s.get("page_end", p_start)
            page_str = f"Trang {p_start}" if p_start == p_end else f"Trang {p_start} - {p_end}"
            score = s.get("score")
            score_str = f" • Điểm phù hợp Rerank: `{score:.3f}`" if score is not None else ""

            st.markdown(f"#### `[{cid}]` {doc}")
            sub_headers = []
            if clause:
                sub_headers.append(f"**{clause}**")
            if heading:
                sub_headers.append(f"*{heading}*")
            if sub_headers:
                st.markdown(" ➔ ".join(sub_headers))

            st.caption(f"📍 {page_str}{score_str}")

            # Đoạn văn bản gốc từ tài liệu (nguyên văn RAG chunk)
            chunk_text = s.get("text") or s.get("snippet", "")
            with st.container(border=True):
                st.markdown(f"**Nội dung văn bản được hệ thống truy hồi đối chiếu:**\n\n{chunk_text}")
            st.divider()


# Hiển thị các tin nhắn cũ
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            display_sources(msg["sources"], expanded=False)

# Ô nhập câu hỏi của người dùng
if prompt := st.chat_input("Nhập câu hỏi của bạn (ví dụ: điều kiện tốt nghiệp, thủ tục học lại, miễn giảm học phí...):"):
    # Hiển thị tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Xử lý sinh câu trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu quy chế và tổng hợp câu trả lời..."):
            res = pipeline.query(
                question=prompt,
                method=retriever_mode,
                top_k=top_k,
                use_rerank=use_rerank
            )
            answer_text = res["answer"]
            sources = res.get("sources", [])
            lat = res.get("latency_ms", {})

            st.markdown(answer_text)

            # Thẻ thông tin độ trễ
            st.caption(
                f"⏱️ Phản hồi trong {lat.get('total_ms', 0):.0f}ms "
                f"(Truy xuất: {lat.get('retrieve_ms', 0):.0f}ms, "
                f"Rerank: {lat.get('rerank_ms', 0):.0f}ms, "
                f"LLM: {lat.get('generate_ms', 0):.0f}ms)"
            )

            # Hiển thị nguồn trích dẫn & văn bản gốc
            if sources:
                display_sources(sources, expanded=True)

            # Lưu lại vào lịch sử
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources": sources
            })
