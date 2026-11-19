"""Embed toàn bộ chunk bằng bge-m3 và lưu vào Qdrant (local data/index, hoặc server nếu có QDRANT_URL).

Chạy: python -m src.ingest.build_index
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from src.config import COLLECTION_NAME, EMBEDDING_MODEL, INDEX_DIR, QDRANT_URL
from src.ingest.chunking import load_chunks


def qdrant_location() -> dict[str, str]:
    """Tham số kết nối Qdrant dùng chung cho build và truy vấn: server mode nếu có QDRANT_URL, không thì local."""
    return {"url": QDRANT_URL} if QDRANT_URL else {"path": str(INDEX_DIR)}


def get_embeddings() -> HuggingFaceEmbeddings:
    """Model embedding dùng chung cho lúc build index và lúc truy vấn."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index() -> QdrantVectorStore:
    """Tạo lại collection từ đầu (force_recreate) để build lại không bị nhân đôi chunk."""
    chunks = load_chunks()
    if not QDRANT_URL:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return QdrantVectorStore.from_documents(
        chunks,
        get_embeddings(),
        **qdrant_location(),
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )


if __name__ == "__main__":
    store = build_index()
    count = store.client.count(COLLECTION_NAME).count
    print(f"Đã lưu {count} chunk vào {QDRANT_URL or INDEX_DIR}")
    for doc in store.similarity_search("Thuật toán A* là gì?", k=3):
        print(f"- {doc.metadata['source']} > {doc.metadata['section']}")
    store.client.close()  # đóng trước khi Python tắt, tránh ImportError trong __del__
