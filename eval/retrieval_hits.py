"""Đo chất lượng retrieval trên bộ dev, không gọi Gemini.

- File hit@k:    mọi file cần có đều nằm trong top-k.
- Section hit@k: mọi (file, mục) cần có đều nằm trong top-k (chặt hơn: đúng mục chứa đáp án).
- MRR@k:         trung bình 1/hạng của chunk đúng mục đầu tiên (0 nếu không có trong top-k).
- Top-1 score:   so với MIN_SCORE để xem ngưỡng từ chối có tách được câu ngoài phạm vi không.

Chạy: python -m eval.retrieval_hits
"""
from transformers.utils import logging as hf_logging

from eval.run_eval import expected_files, load_questions
from src.config import MIN_SCORE, TOP_K
from src.retrieval.search import get_store, search

K_VALUES = (4, 6, 8, 10)


def expected_pairs(question: dict) -> list[tuple[str, str]]:
    """(file, mục) cần có; section là chuỗi hoặc list song song với source_file."""
    section = question["section"]
    sections = section if isinstance(section, list) else [section]
    return list(zip(expected_files(question), sections))


def rank_of(target, items: list) -> int | None:
    """Hạng (bắt đầu từ 1) của lần xuất hiện đầu tiên, None nếu không có."""
    return items.index(target) + 1 if target in items else None


def is_hit(ranks: list[int | None], k: int) -> bool:
    return all(rank is not None and rank <= k for rank in ranks)


def reciprocal_rank(ranks: list[int | None]) -> float:
    found = [rank for rank in ranks if rank is not None]
    return 1 / min(found) if found else 0.0


def main() -> None:
    hf_logging.disable_progress_bar()
    questions = load_questions("dev")
    in_scope = [q for q in questions if expected_files(q)]
    file_hits = {k: 0 for k in K_VALUES}
    section_hits = {k: 0 for k in K_VALUES}
    rr_total = 0.0
    top1_in, top1_out = [], []
    try:
        for q in questions:
            results = search(q["question"], k=max(K_VALUES))
            top1 = results[0][1]
            if not expected_files(q):
                top1_out.append((q["id"], top1))
                continue
            top1_in.append((q["id"], top1))
            pairs = [(doc.metadata["source"], doc.metadata["section"]) for doc, _ in results]
            file_ranks = [rank_of(f, [p[0] for p in pairs]) for f in expected_files(q)]
            section_ranks = [rank_of(p, pairs) for p in expected_pairs(q)]
            for k in K_VALUES:
                file_hits[k] += is_hit(file_ranks, k)
                section_hits[k] += is_hit(section_ranks, k)
            rr_total += reciprocal_rank(section_ranks)
            if not is_hit(section_ranks, TOP_K):
                print(f"q{q['id']:02d} hạng file={file_ranks} hạng mục={section_ranks}  {q['question'][:55]}")
    finally:
        get_store().client.close()  # tránh ImportError trong __del__ khi Python tắt

    n = len(in_scope)
    print(f"\n{'':<10}" + "".join(f"hit@{k:<6}" for k in K_VALUES))
    for name, hits in (("file", file_hits), ("section", section_hits)):
        print(f"{name:<10}" + "".join(f"{100 * hits[k] / n:>5.0f}%   " for k in K_VALUES))
    print(f"\nMRR@{max(K_VALUES)} (section): {rr_total / n:.3f}   ({n} câu trong phạm vi, TOP_K hiện tại = {TOP_K})")
    print(f"Top-1 thấp nhất, câu trong phạm vi: q{min(top1_in, key=lambda x: x[1])[0]} = {min(s for _, s in top1_in):.3f}")
    print(f"Top-1 câu ngoài phạm vi: {', '.join(f'q{i}={s:.3f}' for i, s in top1_out)}   (MIN_SCORE = {MIN_SCORE})")


if __name__ == "__main__":
    main()
