"""Ghép prompt (câu hỏi + chunk) → gọi Gemini → trả lời kèm nguồn."""
import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import LLM_MODEL, MIN_SCORE, NOT_FOUND_ANSWER, ROOT_DIR, TOP_K
from src.retrieval.search import search

SYSTEM_PROMPT = f"""Bạn là trợ giảng môn CS106 - Trí tuệ nhân tạo.
Chỉ trả lời dựa trên các đoạn tài liệu được cung cấp, không dùng kiến thức bên ngoài.
Trả lời bằng tiếng Việt, ngắn gọn, đúng trọng tâm câu hỏi.
Sau mỗi ý, ghi số của đoạn đã dùng trong ngoặc vuông, ví dụ [1] hoặc [2][3].
Nếu các đoạn tài liệu không có thông tin để trả lời, chỉ viết đúng một câu: {NOT_FOUND_ANSWER}"""


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Tạo chat model một lần; báo lỗi rõ nếu thiếu API key."""
    load_dotenv(ROOT_DIR / ".env")
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Thiếu GOOGLE_API_KEY. Thêm dòng GOOGLE_API_KEY=... vào file .env")
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)


def source_label(doc: Document) -> str:
    return f"{doc.metadata['source']} > {doc.metadata['section']}"


def format_context(docs: list[Document]) -> str:
    """Đánh số từng đoạn [1], [2]... để LLM trích lại bằng số."""
    return "\n\n".join(
        f"[{i}] ({source_label(doc)})\n{doc.page_content}" for i, doc in enumerate(docs, start=1)
    )


def cited_sources(answer_text: str, docs: list[Document]) -> list[str]:
    """Nguồn lấy từ metadata (không tin tên file LLM tự viết), chỉ giữ đoạn được trích."""
    if NOT_FOUND_ANSWER.lower() in answer_text.lower():
        return []
    numbers = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
    cited = [doc for i, doc in enumerate(docs, start=1) if i in numbers]
    # LLM quên ghi [n] thì vẫn hiện nguồn: dùng tất cả đoạn đã đưa vào prompt
    return list(dict.fromkeys(source_label(doc) for doc in cited or docs))


def answer(question: str, k: int = TOP_K) -> tuple[str, list[str]]:
    """Trả về (câu trả lời, danh sách nguồn "file > mục"). Từ chối thì danh sách nguồn rỗng."""
    results = search(question, k=k)
    if not results or results[0][1] < MIN_SCORE:
        return NOT_FOUND_ANSWER, []

    docs = [doc for doc, _ in results]
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Tài liệu:\n{format_context(docs)}\n\nCâu hỏi: {question}"),
    ]
    text = get_llm().invoke(messages).text.strip()
    return text, cited_sources(text, docs)
