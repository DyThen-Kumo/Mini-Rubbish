"""Tham số dùng chung cho toàn project. Chỉnh ở đây, không rải trong code."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "OriginalDocs"

# Ingest
CHUNK_SIZE = 1500  # số ký tự tối đa của phần nội dung một chunk (không tính dòng tiêu đề)
