# DANH SÁCH TÀI LIỆU NGUỒN (CORPUS) - DỰ ÁN RAG TIẾNG VIỆT (NLP-21)

> Miền đề tài: **Quy chế đào tạo đại học và quy định liên quan**  
> Định dạng yêu cầu: Cần cả **PDF**, **DOCX**, và **TXT** để kiểm thử 3 nhánh trích xuất (extract pipeline).

---

## 1. Bảng kê tài liệu thực tế

| STT | doc_id (Tên tệp không dấu) | Tên văn bản / Mô tả | Định dạng | Nguồn tải (URL) | Ngày truy cập | SHA-256 Checksum |
|---|---|---|---|---|---|---|
| 1 | `thong_tu_08_2021_bo_gddt.pdf` | Thông tư 08/2021/TT-BGDĐT ban hành Quy chế đào tạo trình độ đại học | PDF | [Cổng Thông tin điện tử Bộ GD&ĐT / Thư Viện Pháp Luật](https://moet.gov.vn) | YYYY-MM-DD | *(Sẽ sinh tự động)* |
| 2 | `quy_che_dao_tao_dh_truong_a.pdf` | Quy chế đào tạo trình độ đại học của Trường ĐH (bản công khai) | PDF | Website phòng Đào tạo trường ĐH | YYYY-MM-DD | *(Sẽ sinh tự động)* |
| 3 | `quy_che_cong_tac_sinh_vien.docx` | Quy định về công tác sinh viên và đánh giá điểm rèn luyện | DOCX | Website trường ĐH / Cổng CTSV | YYYY-MM-DD | *(Sẽ sinh tự động)* |
| 4 | `huong_dan_hoc_lai_tot_nghiep.txt` | Hướng dẫn đăng ký học lại, học cải thiện và điều kiện tốt nghiệp | TXT | Website phòng Đào tạo trường ĐH | YYYY-MM-DD | *(Sẽ sinh tự động)* |
| 5 | `quy_dinh_hoc_phi_mien_giam.pdf` | Quy định về học phí, miễn giảm học phí và chế độ chính sách | PDF | Cổng thông tin sinh viên | YYYY-MM-DD | *(Sẽ sinh tự động)* |

---

## 2. Tiêu chí chọn tài liệu (Theo CHECKLIST §0 & §3)
- [ ] Tên tệp viết thường, không dấu, dùng dấu gạch dưới (VD: `thong_tu_08_2021.pdf`).
- [ ] Là văn bản pháp quy công khai (không vướng bản quyền hoặc tài liệu mật).
- [ ] Phải có cấu trúc rõ ràng: Chương / Điều / Khoản để kiểm nghiệm thuật toán Chunking S3.
- [ ] Phải là **PDF dạng văn bản (Text PDF)** có thể bôi đen copy chữ được. Nếu là PDF dạng scan (ảnh), hệ thống sẽ cảnh báo `needs_ocr=True`.
- [ ] Đặt tất cả các file này vào thư mục: `data/raw/`.
