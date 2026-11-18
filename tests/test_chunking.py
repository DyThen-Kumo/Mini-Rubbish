import pytest

from eval.retrieval_hits import expected_pairs
from eval.run_eval import load_questions
from src.config import CHUNK_SIZE, DOCS_DIR
from src.ingest.chunking import (
    PREAMBLE_TITLE,
    chunk_file,
    load_chunks,
    split_long_text,
    split_sections,
)

SEP = "=" * 60

SAMPLE = f"""TIÊU ĐỀ FILE
Giới thiệu ngắn.

{SEP}
1. MỤC MỘT
{SEP}

Nội dung mục một.

{SEP}
PHẦN II — GỘP
{SEP}

2. MỤC HAI
{SEP}

Nội dung mục hai.
Dòng cuối.
{SEP}
END OF SUMMARY
{SEP}
"""


def test_split_sections_titles_and_bodies():
    sections = split_sections(SAMPLE)
    assert sections == [
        (PREAMBLE_TITLE, "TIÊU ĐỀ FILE\nGiới thiệu ngắn."),
        ("1. MỤC MỘT", "Nội dung mục một."),
        ("2. MỤC HAI", "Nội dung mục hai.\nDòng cuối."),
    ]


def test_split_long_text_respects_size_and_keeps_words():
    text = "\n\n".join(f"Đoạn {i} " + "chữ " * 50 for i in range(20))
    pieces = split_long_text(text, 300)
    assert len(pieces) > 1
    assert all(len(piece) <= 300 for piece in pieces)
    assert " ".join(pieces).split() == text.split()


def test_split_long_text_hard_cuts_unbroken_text():
    pieces = split_long_text("x" * 250, 100)
    assert [len(piece) for piece in pieces] == [100, 100, 50]


def test_real_documents_chunks_have_metadata_and_size():
    chunks = load_chunks()
    sources = {chunk.metadata["source"] for chunk in chunks}
    assert sources == {path.name for path in DOCS_DIR.glob("*.txt")}
    for chunk in chunks:
        header, body = chunk.page_content.split("\n", 1)
        assert chunk.metadata["section"] in header
        assert 0 < len(body) <= CHUNK_SIZE
    sections = {(c.metadata["source"], c.metadata["section"]) for c in chunks}
    assert ("04-05_CSP_and_Local_Search_Summary.txt", "1. CSP LÀ GÌ?") in sections
    assert ("08_MDPs_Summary.txt", "3. MARKOV PROPERTY") in sections


def test_load_chunks_fails_clearly_when_no_documents(tmp_path):
    """Sai DOCS_DIR hoặc thư mục rỗng phải báo lỗi rõ, không build ra index rỗng rồi trả lời "không tìm thấy" mọi câu."""
    with pytest.raises(FileNotFoundError, match="Không tìm thấy file .txt"):
        load_chunks(tmp_path)


def test_chunk_file_handles_bom_header_and_long_sections(tmp_path):
    """File lưu bằng Notepad thường có BOM ở đầu; nếu đọc sai, dòng "=====" đầu tiên không khớp regex và mục 1 bị nhập
    vào "Mở đầu". Mục dài cắt thành nhiều chunk thì chunk nào cũng phải giữ tên mục, để trích nguồn và hit@k còn đúng."""
    path = tmp_path / "99_Demo.txt"
    body = "\n\n".join(f"Đoạn {i} nói về heuristic." for i in range(30))
    path.write_text(f"{SEP}\n1. MỤC DÀI\n{SEP}\n\n{body}\n", encoding="utf-8-sig")

    chunks = chunk_file(path, chunk_size=200)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata == {"source": "99_Demo.txt", "section": "1. MỤC DÀI"}
        assert chunk.page_content.startswith("[99_Demo > 1. MỤC DÀI]\n")
        assert "﻿" not in chunk.page_content


def test_eval_labels_match_real_sections():
    """Nhãn (file, mục) của bộ dev phải trùng tên mục mà chunking sinh ra. Gõ sai một ký tự (dấu, khoảng trắng, "—")
    thì section hit@k tính trượt dù retrieval đúng, làm sai kết luận khi tinh chỉnh. Chỉ đọc bộ dev, không đụng eval/test/."""
    real_pairs = {(c.metadata["source"], c.metadata["section"]) for c in load_chunks()}

    wrong = [(q["id"], pair) for q in load_questions("dev") for pair in expected_pairs(q) if pair not in real_pairs]

    assert wrong == []
