"""Tìm top-k chunk liên quan đến câu hỏi trong index đã build ở Task 4."""
from functools import lru_cache

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from src.config import COLLECTION_NAME, INDEX_DIR, TOP_K
from src.ingest.build_index import get_embeddings


@lru_cache(maxsize=1)
def get_store() -> QdrantVectorStore:
    """Mở index một lần rồi dùng lại: Qdrant local mode khoá thư mục, không cho mở client thứ hai."""
    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"Chưa có index ở {INDEX_DIR}. Chạy: python -m src.ingest.build_index"
        )
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(),
        path=str(INDEX_DIR),
        collection_name=COLLECTION_NAME,
    )


def search(question: str, k: int = TOP_K) -> list[tuple[Document, float]]:
    """Trả về k cặp (chunk, điểm cosine), điểm cao nhất trước. Metadata có `source` và `section`."""
    return get_store().similarity_search_with_score(question, k=k)
