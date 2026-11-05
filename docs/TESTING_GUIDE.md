# 🛠️ HƯỚNG DẪN TEST VÀ CHẠY DỰ ÁN DÀNH CHO MÁY CÓ GPU (NVIDIA)

Tài liệu này dùng để gửi cho bạn cùng nhóm (người có máy GPU) hoặc đưa cho AI bên máy đó để chạy kiểm thử dự án từ đầu đến cuối.

---

## 📌 BƯỚC 1: Clone Repo & Cài đặt Môi trường Chuẩn

Mở PowerShell / Terminal trên máy có GPU:

```bash
# 1. Clone repository về máy
git clone https://github.com/BlueCat215/RAG-For-Vietnamese-Documents.git
cd RAG-For-Vietnamese-Documents

# 2. Tạo môi trường ảo Python 3.10 hoặc 3.11
python -m venv .venv

# Kích hoạt venv (trên Windows PowerShell):
.venv\Scripts\Activate.ps1
# (Hoặc trên CMD: .venv\Scripts\activate.bat)

# 3. BẮT BUỘC: Cài PyTorch hỗ trợ CUDA (cho GPU NVIDIA) TRƯỚC:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. Cài đặt toàn bộ các thư viện còn lại:
pip install -r requirements.txt
```

---

## 📌 BƯỚC 2: Chạy Smoke Test Kiểm tra GPU & VRAM

Chạy lệnh kiểm tra xem PyTorch đã nhận GPU NVIDIA chưa:

```bash
python tests/test_environment.py
```

* **Yêu cầu đạt:** Dòng `CUDA Available:` phải là **`True`** và hiển thị tên GPU (ví dụ: RTX 3050) cùng dung lượng VRAM (~4GB).

---

## 📌 BƯỚC 3: Chuẩn bị Dữ liệu vào `data/raw/`

Copy từ **2 đến 5 file tài liệu quy chế** (PDF văn bản, DOCX, hoặc TXT) vào thư mục:
👉 `data/raw/`

*(Lưu ý: Tên file viết thường không dấu, ví dụ: `thong_tu_08_2021.pdf`, `quy_che_dao_tao.pdf`).*

---

## 📌 BƯỚC 4: Chạy Lập Chỉ Mục (Build Indexing)

Chạy script tự động tiền xử lý, cắt đoạn Điều/Khoản và tính vector trên GPU:

```bash
python run_indexing.py
```

* **Kết quả thành công:**
  - Sinh file `data/processed/chunks_structural.jsonl`.
  - Sinh file `data/index/faiss.index` và `data/index/chunks_metadata.json`.
  - Tự động giải phóng GPU VRAM sau khi tạo xong.

---

## 📌 BƯỚC 5: Chạy Thử Nghiệm Hỏi Đáp (Interactive Test)

Chạy console hỏi đáp trực tiếp với trợ lý AI:

```bash
python run_query.py
```

* **Các ca cần test để kiểm tra chất lượng:**
  1. **Ca 1 (Câu hỏi dễ/trực tiếp):** Hỏi một điều khoản có thật trong tài liệu (VD: *"Điều kiện để sinh viên được công nhận tốt nghiệp là gì?"*).
     - $\rightarrow$ Kiểm tra câu trả lời có trích dẫn nguồn `[1]`, `[2]` không.
  2. **Ca 2 (Câu hỏi paraphrase/khác từ vựng):** Hỏi bằng văn nói của sinh viên (VD: *"Học lại môn bị trượt thế nào?"* thay vì *"Đăng ký lại học phần điểm F"*).
     - $\rightarrow$ Kiểm tra Hybrid Search và Reranker có tìm đúng Điều về học lại không.
  3. **Ca 3 (Câu hỏi từ chối - Test ảo giác):** Hỏi điều mà tài liệu hoàn toàn không có (VD: *"Học phí ngành Y đa khoa của trường là bao nhiêu?"* hoặc *"Thời tiết hôm nay thế nào?"*).
     - $\rightarrow$ Hệ thống phải trả lời: *"Tôi không tìm thấy thông tin này trong tài liệu..."* (Tuyệt đối không được bịa số liệu).
