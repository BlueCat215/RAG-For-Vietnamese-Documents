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

# Hiển thị các tin nhắn cũ
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Xem nguồn trích dẫn & căn cứ pháp lý"):
                for s in msg["sources"]:
                    st.markdown(f"**[{s['citation_id']}] {s['doc_id']}** - *{s['heading']}*")
                    if s.get("clause"):
                        st.markdown(f"📌 **{s['clause']}**")
                    st.caption(f"Trang: {s['page_start']} - Điểm phù hợp: {s['score']:.3f}")
                    st.code(s["snippet"], language="text")

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

            # Hiển thị nguồn trích dẫn
            if sources:
                with st.expander("📚 Xem nguồn trích dẫn & căn cứ pháp lý"):
                    for s in sources:
                        st.markdown(f"**[{s['citation_id']}] {s['doc_id']}** - *{s['heading']}*")
                        if s.get("clause"):
                            st.markdown(f"📌 **{s['clause']}**")
                        st.caption(f"Trang: {s['page_start']} - Điểm phù hợp: {s['score']:.3f}")
                        st.code(s["snippet"], language="text")

            # Lưu lại vào lịch sử
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources": sources
            })
