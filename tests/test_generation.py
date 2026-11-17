"""Test phần generation, không gọi Gemini (dùng LLM giả) và không mở index (search giả)."""
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel

from src.config import NOT_FOUND_ANSWER
from src.generation import answer as generation

DOCS = [
    Document("A* dùng f(n) = g(n) + h(n).", metadata={"source": "03_Informed.txt", "section": "2. A*"}),
    Document("Heuristic admissible.", metadata={"source": "03_Informed.txt", "section": "2. A*"}),
    Document("UCS mở rộng theo chi phí.", metadata={"source": "02_Uninformed.txt", "section": "4. UCS"}),
]


def test_format_context_numbers_chunks_with_source():
    context = generation.format_context(DOCS)

    assert context.startswith("[1] (03_Informed.txt > 2. A*)\nA* dùng")
    assert "[3] (02_Uninformed.txt > 4. UCS)" in context


def test_cited_sources_keeps_only_cited_and_dedups():
    sources = generation.cited_sources("A* dùng f = g + h [1][2].", DOCS)

    assert sources == ["03_Informed.txt > 2. A*"]


def test_cited_sources_falls_back_to_all_when_no_citation():
    sources = generation.cited_sources("A* dùng f = g + h.", DOCS)

    assert sources == ["03_Informed.txt > 2. A*", "02_Uninformed.txt > 4. UCS"]


def test_cited_sources_empty_when_not_found():
    assert generation.cited_sources(f"{NOT_FOUND_ANSWER}.", DOCS) == []


def test_answer_refuses_below_min_score_without_calling_llm(monkeypatch):
    monkeypatch.setattr(generation, "search", lambda question, k: [(DOCS[0], 0.1)])
    monkeypatch.setattr(generation, "get_llm", lambda: 1 / 0)  # gọi LLM là test lỗi

    assert generation.answer("Hôm nay ăn gì?") == (NOT_FOUND_ANSWER, [])


def test_answer_returns_llm_text_and_cited_sources(monkeypatch):
    fake_llm = FakeListChatModel(responses=["f(n) = g(n) + h(n) [1]."])
    monkeypatch.setattr(generation, "search", lambda question, k: [(doc, 0.7) for doc in DOCS])
    monkeypatch.setattr(generation, "get_llm", lambda: fake_llm)

    text, sources = generation.answer("A* là gì?")

    assert text == "f(n) = g(n) + h(n) [1]."
    assert sources == ["03_Informed.txt > 2. A*"]
