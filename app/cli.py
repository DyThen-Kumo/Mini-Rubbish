"""CLI hỏi đáp tài liệu CS106.

Chạy: python -m app.cli          (thêm --debug để in điểm cosine của các chunk)
"""
import argparse
import warnings

from langchain_google_genai.chat_models import GoogleAPIError, GoogleRateLimitError
from transformers.utils import logging as hf_logging

from src.config import MIN_SCORE
from src.generation.answer import answer
from src.retrieval.search import get_store, search

EXIT_COMMANDS = {"exit", "quit", "thoát"}


def format_reply(text: str, sources: list[str]) -> str:
    """Câu trả lời, rồi danh sách nguồn (nếu có)."""
    lines = [text]
    if sources:
        lines += ["", "Nguồn:"] + [f"  - {source}" for source in sources]
    return "\n".join(lines)


def format_debug(question: str) -> str:
    """Điểm cosine của các chunk tìm được, để so với MIN_SCORE."""
    # shortcut: gọi search() lần hai (answer() đã gọi), tốn thêm một lần embed câu hỏi; chỉ chạy khi --debug
    results = search(question)
    lines = [f"[debug] MIN_SCORE = {MIN_SCORE}"]
    lines += [
        f"  {score:.3f}  {doc.metadata['source']} > {doc.metadata['section']}"
        for doc, score in results
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hỏi đáp tài liệu CS106")
    parser.add_argument("--debug", action="store_true", help="in điểm cosine của các chunk tìm được")
    args = parser.parse_args()

    hf_logging.disable_progress_bar()  # ẩn thanh "Loading weights" khi nạp bge-m3
    warnings.filterwarnings("ignore", message=".*sampling parameter.*")  # Flash Lite bỏ qua temperature

    print("Hỏi đáp CS106. Gõ 'exit' để thoát.")
    try:
        while True:
            question = input("\nCâu hỏi: ").strip()
            if not question:
                continue
            if question.lower() in EXIT_COMMANDS:
                break
            if args.debug:
                print(format_debug(question))
            try:
                text, sources = answer(question)
            except (GoogleAPIError, GoogleRateLimitError) as error:
                print(f"Gemini đang quá tải hoặc hết quota, thử lại sau. ({str(error).split('.')[0]})")
                continue
            print(format_reply(text, sources))
    except (KeyboardInterrupt, EOFError):
        print()
    except (FileNotFoundError, RuntimeError) as error:  # chưa build index / thiếu GOOGLE_API_KEY
        raise SystemExit(f"Lỗi: {error}")
    finally:
        if get_store.cache_info().currsize:
            get_store().client.close()  # đóng trước khi Python tắt, tránh ImportError trong __del__


if __name__ == "__main__":
    main()
