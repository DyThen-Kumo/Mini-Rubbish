from src.config import CHUNK_SIZE, DOCS_DIR
from src.ingest.chunking import (
    PREAMBLE_TITLE,
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
