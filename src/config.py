"""Tham số dùng chung cho toàn project. Chỉnh ở đây, không rải trong code."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "OriginalDocs"

# Ingest
CHUNK_SIZE = 1500  # số ký tự tối đa của phần nội dung một chunk (không tính dòng tiêu đề)

# Index
EMBEDDING_MODEL = "BAAI/bge-m3"
INDEX_DIR = ROOT_DIR / "data" / "index"
COLLECTION_NAME = "cs106"
QDRANT_URL = os.getenv("QDRANT_URL")  # có (Docker Compose đặt) thì dùng Qdrant server; không thì local mode INDEX_DIR

# Retrieval
TOP_K = 4  # số chunk đưa vào prompt

# Generation
LLM_MODEL = "gemini-3.5-flash-lite"
GEMINI_RPM = 10  # giới hạn request/phút của free tier cho LLM_MODEL; eval tự giãn cách theo số này
MIN_SCORE = 0.5  # chunk tốt nhất dưới ngưỡng này thì từ chối luôn, không gọi LLM
NOT_FOUND_ANSWER = "Không tìm thấy trong tài liệu"  # phải khớp expected_answer trong eval/
