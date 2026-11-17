"""Chấm hệ thống trên eval/questions: độ chính xác, nguồn, từ chối, trung thực, retrieval hit@k.

Chạy:      python -m eval.run_eval
Chạy tiếp: python -m eval.run_eval --resume eval/results/<file>.jsonl
"""
import argparse
import json
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_google_genai.chat_models import GoogleAPIError, GoogleRateLimitError
from pydantic import BaseModel
from transformers.utils import logging as hf_logging

from src.config import GEMINI_RPM, NOT_FOUND_ANSWER, ROOT_DIR
from src.generation.answer import answer, format_context, get_llm
from src.retrieval.search import get_store, search

QUESTIONS_DIR = ROOT_DIR / "eval" / "questions"
RESULTS_DIR = ROOT_DIR / "eval" / "results"

JUDGE_PROMPT = """Bạn là giám khảo nghiêm khắc chấm hệ thống hỏi đáp môn CS106.
Bạn nhận: câu hỏi, đáp án mẫu, các đoạn tài liệu hệ thống đã đọc, và câu trả lời của hệ thống.
Bước 1: tách đáp án mẫu thành các ý chính. Mỗi định nghĩa, công thức, điều kiện, tính chất, vế so sánh là một ý.
Bước 2: kiểm tra từng ý chính có trong câu trả lời không. Diễn đạt khác vẫn tính là có; nhắc tên mà thiếu công thức hoặc điều kiện thì tính là thiếu.
Trả về:
- missing_points: các ý chính của đáp án mẫu mà câu trả lời không nêu. Danh sách rỗng nếu đủ.
- wrong_points: các ý trong câu trả lời mâu thuẫn với đáp án mẫu hoặc tài liệu. Danh sách rỗng nếu không có.
- faithful: true nếu mọi ý trong câu trả lời đều có trong các đoạn tài liệu.
- reason: một câu ngắn bằng tiếng Việt giải thích nhận xét.
Không cộng điểm vì câu trả lời dài hay trình bày đẹp."""


class Verdict(BaseModel):
    missing_points: list[str]
    wrong_points: list[str]
    faithful: bool
    reason: str

    @property
    def correct(self) -> bool:
        """Đúng khi không thiếu ý chính nào và không có ý sai (quyết định ở code, không để judge tự chọn)."""
        return not self.missing_points and not self.wrong_points


_last_call = 0.0


def wait_for_rate_limit() -> None:
    """Giãn cách các lần gọi Gemini để không vượt GEMINI_RPM."""
    global _last_call
    delay = _last_call + 60 / GEMINI_RPM - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    _last_call = time.monotonic()


def load_questions() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(QUESTIONS_DIR.glob("q*.json"))]


def expected_files(question: dict) -> list[str]:
    """source_file có thể là null (ngoài phạm vi), một tên file, hoặc list (câu cần nhiều bài)."""
    source = question["source_file"]
    if source is None:
        return []
    return source if isinstance(source, list) else [source]


def is_refusal(answer_text: str) -> bool:
    return NOT_FOUND_ANSWER.lower() in answer_text.lower()


def score(
    question: dict, answer_text: str, sources: list[str], top_files: list[str], verdict: Verdict | None
) -> dict:
    """Chấm một câu. Câu nhiều bài phải có đủ mọi file mới tính nguồn/hit đúng."""
    files = expected_files(question)
    refused = is_refusal(answer_text)
    cited_files = {source.split(" > ")[0] for source in sources}
    if files:
        correct = not refused and verdict is not None and verdict.correct
        source_ok = set(files) <= cited_files
        retrieval_hit = set(files) <= set(top_files)
    else:
        correct, source_ok, retrieval_hit = refused, not sources, None
    return {
        "id": question["id"],
        "type": question.get("type", "thường"),
        "question": question["question"],
        "expected_files": files,
        "expected_answer": question["expected_answer"],
        "answer": answer_text,
        "sources": sources,
        "refused": refused,
        "correct": correct,
        "source_ok": source_ok,
        "retrieval_hit": retrieval_hit,
        "faithful": verdict.faithful if verdict else None,
        "missing_points": verdict.missing_points if verdict else [],
        "wrong_points": verdict.wrong_points if verdict else [],
        "reason": verdict.reason if verdict else "",
        "answer_len": len(answer_text),
    }


def pct(count: int, total: int) -> str:
    return f"{count}/{total} ({100 * count / total:.0f}%)" if total else "0/0"


def summarize(records: list[dict]) -> str:
    in_scope = [r for r in records if r["expected_files"]]
    out_scope = [r for r in records if not r["expected_files"]]
    answered = [r for r in records if not r["refused"]]
    judged = [r for r in records if r["faithful"] is not None]
    lines = [
        f"Độ chính xác:          {pct(sum(r['correct'] for r in records), len(records))}  (mục tiêu >= 80%)",
        f"Nguồn đúng:            {pct(sum(r['source_ok'] for r in in_scope), len(in_scope))}  (câu trong phạm vi)",
        f"Câu trả lời có nguồn:  {pct(sum(bool(r['sources']) for r in answered), len(answered))}",
        f"Từ chối đúng:          {pct(sum(r['refused'] for r in out_scope), len(out_scope))}  (câu ngoài phạm vi)",
        f"Từ chối oan:           {sum(r['refused'] for r in in_scope)} câu trong phạm vi",
        f"Trung thực:            {pct(sum(r['faithful'] for r in judged), len(judged))}  (câu được judge chấm)",
        f"Retrieval hit@k:       {pct(sum(r['retrieval_hit'] for r in in_scope), len(in_scope))}",
        f"Độ dài trả lời TB:     {sum(r['answer_len'] for r in records) // max(len(records), 1)} ký tự",
        "",
        "Độ chính xác theo loại câu hỏi:",
    ]
    by_type = defaultdict(list)
    for record in records:
        by_type[record["type"]].append(record)
    lines += [f"  {kind:<15} {pct(sum(r['correct'] for r in rs), len(rs))}" for kind, rs in by_type.items()]
    lines.append(f"\nCâu sai (cần duyệt tay): {[r['id'] for r in records if not r['correct']]}")
    return "\n".join(lines)


def judge(question: dict, answer_text: str, docs: list[Document]) -> Verdict:
    wait_for_rate_limit()
    prompt = (
        f"Câu hỏi: {question['question']}\n\n"
        f"Đáp án mẫu: {question['expected_answer']}\n\n"
        f"Tài liệu hệ thống đã đọc:\n{format_context(docs)}\n\n"
        f"Câu trả lời của hệ thống: {answer_text}"
    )
    return get_llm().with_structured_output(Verdict).invoke([("system", JUDGE_PROMPT), ("human", prompt)])


def evaluate(question: dict) -> dict:
    # shortcut: search() chạy hai lần (ở đây và trong answer()) để lấy top-k và context cho judge; chỉ tốn thêm một lần embed
    docs = [doc for doc, _ in search(question["question"])]
    wait_for_rate_limit()
    answer_text, sources = answer(question["question"])
    verdict = None
    if expected_files(question) and not is_refusal(answer_text):
        verdict = judge(question, answer_text, docs)
    return score(question, answer_text, sources, [doc.metadata["source"] for doc in docs], verdict)


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="Chấm hệ thống trên eval/questions")
    parser.add_argument("--resume", type=Path, help="file .jsonl của lần chạy bị dừng giữa chừng")
    args = parser.parse_args()

    hf_logging.disable_progress_bar()
    warnings.filterwarnings("ignore", message=".*sampling parameter.*")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = args.resume or RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    done = {record["id"] for record in read_records(path)}
    todo = [q for q in load_questions() if q["id"] not in done]
    print(f"Ghi kết quả vào {path} ({len(done)} câu đã có, còn {len(todo)} câu)")

    try:
        for i, question in enumerate(todo, start=1):
            try:
                record = evaluate(question)
            except (GoogleAPIError, GoogleRateLimitError) as error:
                print(f"\nGemini lỗi ở câu {question['id']}: {str(error).split('.')[0]}")
                print(f"Chạy tiếp: python -m eval.run_eval --resume {path}")
                return
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(todo)}] q{question['id']:02d} {'đúng' if record['correct'] else 'SAI'}")
    finally:
        if get_store.cache_info().currsize:
            get_store().client.close()  # đóng trước khi Python tắt, tránh ImportError trong __del__

    summary = summarize(sorted(read_records(path), key=lambda r: r["id"]))
    path.with_suffix(".summary.txt").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
