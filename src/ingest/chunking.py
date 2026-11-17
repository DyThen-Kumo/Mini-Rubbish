"""Đọc tài liệu gốc và tách thành chunk theo mục "=====", mục dài thì cắt tiếp."""
import re
from pathlib import Path

from langchain_core.documents import Document

from src.config import CHUNK_SIZE, DOCS_DIR

PREAMBLE_TITLE = "Mở đầu"

# Khối tiêu đề: dòng "=====", (có thể 1 dòng trống), 1 dòng tiêu đề, rồi dòng "=====".
# Dòng "=====" đóng chỉ được nhìn trước (lookahead), không bị nuốt, vì trong file
# 04-05 nó đồng thời là dòng mở của tiêu đề kế tiếp.
SECTION_PATTERN = re.compile(
    r"^={10,}[ \t]*\n(?:[ \t]*\n)?(?!=)(\S.*?)[ \t]*\n(?=={10,}[ \t]*$)",
    re.MULTILINE,
)
SEPARATOR_LINE = re.compile(r"^={10,}.*\n?", re.MULTILINE)
SPLIT_SEPARATORS = ("\n\n", "\n", " ")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Tách text thành các cặp (tiêu đề mục, nội dung); bỏ mục rỗng."""
    matches = list(SECTION_PATTERN.finditer(text))
    first_start = matches[0].start() if matches else len(text)
    raw_sections = [(PREAMBLE_TITLE, text[:first_start])]
    for current, following in zip(matches, matches[1:] + [None]):
        end = following.start() if following else len(text)
        raw_sections.append((current.group(1), text[current.end():end]))

    sections = []
    for title, body in raw_sections:
        body = SEPARATOR_LINE.sub("", body).strip()
        if body:
            sections.append((title, body))
    return sections


def split_long_text(
    text: str, chunk_size: int, separators: tuple[str, ...] = SPLIT_SEPARATORS
) -> list[str]:
    """Cắt text thành các đoạn <= chunk_size, ưu tiên cắt ở đoạn văn, rồi dòng, rồi từ."""
    # shortcut: không có overlap giữa các chunk con; dòng tiêu đề mục ở đầu mỗi chunk
    # giữ ngữ cảnh. Thêm overlap nếu eval (Task 8) cho thấy mất ý ở ranh giới cắt.
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, finer_separators = separators[0], separators[1:]
    pieces, current = [], ""
    for part in text.split(separator):
        candidate = f"{current}{separator}{part}" if current else part
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if len(part) <= chunk_size:
            current = part
        else:
            pieces.extend(split_long_text(part, chunk_size, finer_separators))
            current = ""
    if current:
        pieces.append(current)
    return [piece.strip() for piece in pieces if piece.strip()]


def chunk_file(path: Path, chunk_size: int = CHUNK_SIZE) -> list[Document]:
    """Đọc một file tài liệu và trả về các chunk kèm metadata nguồn."""
    text = path.read_text(encoding="utf-8-sig")
    documents = []
    for title, body in split_sections(text):
        for piece in split_long_text(body, chunk_size):
            documents.append(
                Document(
                    # Tiêu đề ở đầu để vector của chunk con vẫn "biết" nó thuộc mục nào.
                    page_content=f"[{path.stem} > {title}]\n{piece}",
                    metadata={"source": path.name, "section": title},
                )
            )
    return documents


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """Tách toàn bộ file .txt trong docs_dir thành chunk."""
    paths = sorted(docs_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy file .txt nào trong {docs_dir}")
    return [chunk for path in paths for chunk in chunk_file(path)]
