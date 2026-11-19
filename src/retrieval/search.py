"""Tìm top-k chunk liên quan đến câu hỏi trong index đã build ở Task 4."""
from functools import lru_cache

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.config import COLLECTION_NAME, INDEX_DIR, QDRANT_URL, TOP_K
from src.ingest.build_index import get_embeddings, qdrant_location

BUILD_HINT = "Chạy: python -m src.ingest.build_index"


@lru_cache(maxsize=1)
def get_store() -> QdrantVectorStore:
    """Mở index một lần rồi dùng lại: Qdrant local mode khoá thư mục, không cho mở client thứ hai."""
    # Kiểm tra thư mục trước khi mở client: local mode tự tạo thư mục rỗng nếu chưa có
    if not QDRANT_URL and not INDEX_DIR.exists():
        raise FileNotFoundError(f"Chưa có index ở {INDEX_DIR}. {BUILD_HINT}")
    client = QdrantClient(**qdrant_location())
    if not client.collection_exists(COLLECTION_NAME):
        client.close()
        raise FileNotFoundError(f"Qdrant chưa có collection '{COLLECTION_NAME}'. {BUILD_HINT}")
    return QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=get_embeddings())


def search(question: str, k: int = TOP_K) -> list[tuple[Document, float]]:
    """Trả về k cặp (chunk, điểm cosine), điểm cao nhất trước. Metadata có `source` và `section`."""
    return get_store().similarity_search_with_score(question, k=k)
