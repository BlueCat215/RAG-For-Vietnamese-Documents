"""
Script chạy thử nghiệm hỏi đáp dòng lệnh (CLI Interactive Test)
Dùng để kiểm tra nhanh RAG Pipeline sau khi đã build index xong.
"""

import sys
import io

# Đảm bảo UTF-8 toàn diện trên môi trường Windows
if sys.platform == "win32":
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.pipeline import RAGPipeline


def main():
    print("=" * 60)
    print("💬 HỆ THỐNG HỎI ĐÁP QUY CHẾ ĐÀO TẠO ĐẠI HỌC (CLI TEST)")
    print("=" * 60)

    # 1. Khởi tạo pipeline
    pipeline = RAGPipeline(config_path="config/config.yaml")
    try:
        pipeline.initialize()
    except Exception as e:
        print(f"[!] Lỗi khi nạp pipeline: {e}")
        print("    👉 Bạn đã chạy lệnh 'python run_indexing.py' để dựng chỉ mục chưa?")
        return

    print("\n[+] Sẵn sàng! Gõ câu hỏi của bạn (hoặc gõ 'q' để thoát):")
    print("-" * 60)

    while True:
        try:
            query = input("\n👤 Sinh viên hỏi: ").strip()
            if not query:
                continue
            
            # Nếu chạy từ pipe stdin (không phải TTY), hiển thị câu hỏi ra console/log
            if not sys.stdin.isatty():
                print(query)

            if query.lower() in ["q", "exit", "quit"]:
                print("Tạm biệt!")
                break

            print("[*] Đang truy xuất dữ liệu & sinh câu trả lời...")
            res = pipeline.query(query, method="hybrid", top_k=5, use_rerank=True)

            print("\n🤖 Trợ lý trả lời:")
            print(res["answer"])

            # Hiển thị độ trễ
            lat = res["latency_ms"]
            print(f"\n⏱️ Thời gian phản hồi: {lat.get('total_ms', 0):.0f}ms (Retrieve: {lat.get('retrieve_ms', 0):.0f}ms, Rerank: {lat.get('rerank_ms', 0):.0f}ms, LLM: {lat.get('generate_ms', 0):.0f}ms)")

            # Hiển thị nguồn trích dẫn
            if res.get("sources"):
                print("\n📚 Nguồn trích dẫn:")
                for s in res["sources"]:
                    print(f"  [{s['citation_id']}] {s['doc_id']} - {s['heading']} (Điểm: {s['score']:.3f})")
            elif res.get("refused"):
                print(f"  ℹ️ Lý do từ chối: {res.get('refusal_reason', 'Không đủ dữ liệu trong context')}")

            print("-" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\nĐã thoát.")
            break
        except Exception as e:
            print(f"[!] Có lỗi xảy ra trong quá trình xử lý: {e}")


if __name__ == "__main__":
    main()
