import pytest

from eval.retrieval_hits import expected_pairs, is_hit, rank_of
from eval.run_eval import expected_files, load_questions
from src.config import COLLECTION_NAME, INDEX_DIR, TOP_K
from src.ingest.build_index import qdrant_location
from src.ingest.chunking import load_chunks
from src.retrieval.search import get_store, search


def test_search_returns_k_chunks_with_metadata_sorted_by_score():
    results = search("Thuật toán A* là gì?", k=3)

    assert len(results) == 3
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    for doc, _ in results:
        assert doc.metadata["source"]
        assert doc.metadata["section"]


def test_search_finds_relevant_lesson():
    results = search("Thuật toán A* là gì?", k=3)

    sources = [doc.metadata["source"] for doc, _ in results]
    assert any("Informed" in source for source in sources)


def test_get_store_fails_clearly_when_index_missing(tmp_path, monkeypatch):
    """Chạy app/eval trước khi build index phải báo lỗi kèm lệnh build, không để Qdrant tự tạo index rỗng.
    Gọi __wrapped__ để bỏ qua cache, không đóng/mở lại index thật mà các test khác đang dùng."""
    monkeypatch.setattr("src.retrieval.search.INDEX_DIR", tmp_path / "missing")

    with pytest.raises(FileNotFoundError, match="python -m src.ingest.build_index"):
        get_store.__wrapped__()


def test_get_store_fails_clearly_when_server_has_no_collection(monkeypatch):
    """Server mode (Docker) mà chưa build index vào Qdrant thì báo lỗi kèm lệnh build, không để lỗi 404 thô.
    Dùng Qdrant in-memory thay cho server thật."""
    monkeypatch.setattr("src.retrieval.search.QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setattr("src.retrieval.search.qdrant_location", lambda: {"location": ":memory:"})

    with pytest.raises(FileNotFoundError, match="python -m src.ingest.build_index"):
        get_store.__wrapped__()


def test_qdrant_location_uses_server_only_when_url_set(monkeypatch):
    """Không có QDRANT_URL thì giữ local mode, để CLI/eval/pytest chạy được mà không cần Docker."""
    monkeypatch.setattr("src.ingest.build_index.QDRANT_URL", None)
    assert qdrant_location() == {"path": str(INDEX_DIR)}

    monkeypatch.setattr("src.ingest.build_index.QDRANT_URL", "http://qdrant:6333")
    assert qdrant_location() == {"url": "http://qdrant:6333"}


def test_index_is_up_to_date_with_documents():
    """Số vector trong index phải bằng số chunk sinh từ tài liệu hiện tại. Đổi CHUNK_SIZE/chunking/tài liệu mà quên build
    lại, hoặc chuyển Qdrant sang server mode (Task 12) mà trỏ nhầm collection, thì test này fail."""
    count = get_store().client.count(COLLECTION_NAME, exact=True).count

    assert count == len(load_chunks())


def test_retrieval_quality_does_not_regress():
    """Section hit@TOP_K trên bộ dev không được tụt dưới 90% (đo ở Task 9: 92%). Lưới an toàn khi đổi embedding,
    chunking, TOP_K hay cách mở Qdrant: biết ngay retrieval có hỏng không mà không cần chạy eval và chấm bằng ChatGPT."""
    questions = [q for q in load_questions("dev") if expected_files(q)]
    hits = 0
    for q in questions:
        pairs = [(doc.metadata["source"], doc.metadata["section"]) for doc, _ in search(q["question"], k=TOP_K)]
        hits += is_hit([rank_of(pair, pairs) for pair in expected_pairs(q)], TOP_K)

    assert hits / len(questions) >= 0.9
