from src.retrieval.search import search


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
