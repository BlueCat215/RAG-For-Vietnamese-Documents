import json
import os
from typing import Any, Dict, List
import yaml


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
  """Load file cấu hình YAML chung cho toàn bộ hệ thống RAG."""
  # Kiểm tra file config có tồn tại không, báo lỗi ngắt chương trình nếu thiếu
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"Không tìm thấy file config tại {config_path}")

  # Mở file với encoding UTF-8, dùng safe_load để bảo mật và tránh lỗ hổng bảo mật từ yaml.load
  with open(config_path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
  """Đọc file định dạng JSONL (JSON Lines) dòng-theo-dòng, tối ưu cho corpus dữ liệu lớn."""
  # Trả về list rỗng nếu file chưa tồn tại (tránh crash pipeline khi chưa sinh dữ liệu)
  if not os.path.exists(file_path):
    return []

  data = []
  # Đọc tuần tự từng dòng giúp tiết kiệm RAM khi xử lý các file dữ liệu hàng trăm MB
  with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      # Bỏ qua các dòng trống/khoảng trắng thừa để tránh lỗi parse JSON
      if line:
        data.append(json.loads(line))

  return data