"""API hỏi đáp tài liệu CS106.

Chạy: uvicorn api.main:app      (một worker: Qdrant local mode khoá thư mục index)
"""
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_google_genai.chat_models import GoogleAPIError, GoogleRateLimitError
from pydantic import BaseModel, StringConstraints

from src.config import DOCS_DIR, ROOT_DIR
from src.generation.answer import answer, get_llm
from src.ingest.chunking import load_chunks
from src.retrieval.search import get_store

INDEX_HTML = ROOT_DIR / "app" / "web" / "index.html"


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class AskResponse(BaseModel):
    answer: str
    sources: list[str]  # "file > mục"; rỗng khi không tìm thấy trong tài liệu


class Lecture(BaseModel):
    source: str
    sections: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Nạp bge-m3 + index và kiểm tra API key lúc start: thiếu thì server báo lỗi ngay, request đầu không phải chờ
    get_store()
    get_llm()
    yield
    get_store().client.close()


app = FastAPI(title="CS106 RAG", lifespan=lifespan)
# StaticFiles chặn sẵn đường dẫn thoát ra ngoài thư mục (../.env); không dùng /docs vì trùng Swagger UI
app.mount("/files", StaticFiles(directory=DOCS_DIR), name="files")  # tải tài liệu .txt từ mục lục
app.mount("/assets", StaticFiles(directory=ROOT_DIR / "assets"), name="assets")  # logo


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/toc")
@lru_cache(maxsize=1)  # tài liệu không đổi khi server đang chạy
def table_of_contents() -> list[Lecture]:
    """Mục lục: từng bài và các mục của nó, theo đúng thứ tự trong tài liệu."""
    toc: dict[str, list[str]] = {}
    for chunk in load_chunks():
        sections = toc.setdefault(chunk.metadata["source"], [])
        if chunk.metadata["section"] not in sections:  # mục dài bị cắt thành nhiều chunk
            sections.append(chunk.metadata["section"])
    return [Lecture(source=source, sections=sections) for source, sections in toc.items()]


@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    """`def` thường (không async): answer() chạy blocking, FastAPI tự đẩy sang threadpool."""
    try:
        text, sources = answer(request.question)
    except GoogleRateLimitError as error:
        raise HTTPException(429, f"Gemini hết quota, thử lại sau. ({error})") from error
    except GoogleAPIError as error:
        raise HTTPException(502, f"Gemini lỗi. ({error})") from error
    except (FileNotFoundError, RuntimeError) as error:  # chưa build index / thiếu GOOGLE_API_KEY
        raise HTTPException(503, str(error)) from error
    return AskResponse(answer=text, sources=sources)
