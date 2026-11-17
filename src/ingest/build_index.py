"""Embed toàn bộ chunk bằng bge-m3 và lưu vào Qdrant local (data/index).

Chạy: python -m src.ingest.build_index
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from src.config import COLLECTION_NAME, EMBEDDING_MODEL, INDEX_DIR
from src.ingest.chunking import load_chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    """Model embedding dùng chung cho lúc build index và lúc truy vấn."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index() -> QdrantVectorStore:
    """Tạo lại collection từ đầu (force_recreate) để build lại không bị nhân đôi chunk."""
    chunks = load_chunks()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return QdrantVectorStore.from_documents(
        chunks,
        get_embeddings(),
        path=str(INDEX_DIR),
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )


if __name__ == "__main__":
    store = build_index()
    count = store.client.count(COLLECTION_NAME).count
    print(f"Đã lưu {count} chunk vào {INDEX_DIR}")
    for doc in store.similarity_search("Thuật toán A* là gì?", k=3):
        print(f"- {doc.metadata['source']} > {doc.metadata['section']}")
    store.client.close()  # đóng trước khi Python tắt, tránh ImportError trong __del__
