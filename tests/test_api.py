"""Test endpoint POST /ask (answer giả: không gọi Gemini, không mở index).

TestClient không dùng `with` nên lifespan (nạp index, kiểm tra key) không chạy.
"""
import pytest
from fastapi.testclient import TestClient
from langchain_google_genai.chat_models import GoogleRateLimitError

from api import main
from src.config import DOCS_DIR, NOT_FOUND_ANSWER

client = TestClient(main.app)


def test_ask_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr(main, "answer", lambda question: ("f = g + h [1].", ["03_Informed.txt > 2. A*"]))

    response = client.post("/ask", json={"question": "  A* là gì?  "})

    assert response.status_code == 200
    assert response.json() == {"answer": "f = g + h [1].", "sources": ["03_Informed.txt > 2. A*"]}


def test_ask_out_of_scope_has_no_sources(monkeypatch):
    monkeypatch.setattr(main, "answer", lambda question: (NOT_FOUND_ANSWER, []))

    response = client.post("/ask", json={"question": "Giá vàng hôm nay?"})

    assert response.json() == {"answer": NOT_FOUND_ANSWER, "sources": []}


@pytest.mark.parametrize("body", [{"question": "   "}, {"question": "x" * 1001}, {}])
def test_ask_rejects_invalid_question(body):
    assert client.post("/ask", json=body).status_code == 422


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (GoogleRateLimitError("quota"), 429),
        (FileNotFoundError("Chưa có index"), 503),
        (RuntimeError("Thiếu GOOGLE_API_KEY"), 503),
    ],
)
def test_ask_maps_errors_to_http_status(monkeypatch, error, status):
    def failing_answer(question):
        raise error

    monkeypatch.setattr(main, "answer", failing_answer)

    response = client.post("/ask", json={"question": "A* là gì?"})

    assert response.status_code == status
    assert str(error) in response.json()["detail"]


def test_toc_lists_every_lecture_with_sections():
    toc = client.get("/toc").json()

    assert len(toc) == 9
    assert all(lecture["sections"] for lecture in toc)
    for lecture in toc:
        assert len(lecture["sections"]) == len(set(lecture["sections"]))  # mục dài nhiều chunk chỉ hiện một lần


def test_home_serves_chat_page():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="toc"' in response.text


def test_files_downloads_original_document():
    """Link "Tải tài liệu" trong mục lục trỏ tới /files/<tên file>; tên có dấu cách phải tải được."""
    name = "09_Reinforcement Learning_Summary.txt"

    response = client.get(f"/files/{name}")

    assert response.status_code == 200
    assert response.content == (DOCS_DIR / name).read_bytes()


def test_files_cannot_escape_docs_dir():
    """Không được lộ file ngoài OriginalDocs/ (ví dụ .env chứa API key)."""
    assert client.get("/files/..%2F.env").status_code == 404
    assert client.get("/files/..%2FCLAUDE.md").status_code == 404


def test_logo_is_served():
    response = client.get("/assets/images/LOGO_Mini_Rubbish_Stelle.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
