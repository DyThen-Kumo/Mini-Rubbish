"""Test phần hiển thị của CLI (search giả, không mở index)."""
from langchain_core.documents import Document

from app import cli
from src.config import NOT_FOUND_ANSWER


def test_format_reply_lists_sources():
    reply = cli.format_reply("f(n) = g(n) + h(n) [1].", ["03_Informed.txt > 2. A*"])

    assert reply == "f(n) = g(n) + h(n) [1].\n\nNguồn:\n  - 03_Informed.txt > 2. A*"


def test_format_reply_without_sources_shows_answer_only():
    assert cli.format_reply(NOT_FOUND_ANSWER, []) == NOT_FOUND_ANSWER


def test_format_debug_shows_scores(monkeypatch):
    doc = Document("A*", metadata={"source": "03_Informed.txt", "section": "2. A*"})
    monkeypatch.setattr(cli, "search", lambda question: [(doc, 0.7412)])

    debug = cli.format_debug("A* là gì?")

    assert "MIN_SCORE" in debug
    assert "0.741  03_Informed.txt > 2. A*" in debug
