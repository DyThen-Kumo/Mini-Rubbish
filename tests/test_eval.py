"""Test phần chấm điểm của eval (không gọi Gemini, không mở index)."""
import pytest

from eval.run_eval import Verdict, apply_verdicts, expected_files, load_questions, score, summarize
from src.config import NOT_FOUND_ANSWER

A_STAR = {"id": 1, "question": "A*?", "expected_answer": "f = g + h", "source_file": "03.txt", "section": "5. A*"}
MULTI = {"id": 2, "question": "So sánh?", "expected_answer": "...", "source_file": ["06.txt", "07.txt"], "section": []}
OUT = {"id": 3, "question": "SVM?", "expected_answer": NOT_FOUND_ANSWER, "source_file": None, "section": None}
GOOD = Verdict(missing_points=[], wrong_points=[], faithful=True, reason="đủ ý")


def test_verdict_is_wrong_when_missing_or_wrong_points():
    assert GOOD.correct
    assert not Verdict(missing_points=["điều kiện cắt"], wrong_points=[], faithful=True, reason="").correct
    assert not Verdict(missing_points=[], wrong_points=["sai công thức"], faithful=True, reason="").correct


def test_apply_verdicts_scores_only_judged_questions():
    in_scope = score(A_STAR, "f = g + h [1]", ["03.txt > 5. A*"], ["03.txt"], None)
    out_scope = score(OUT, NOT_FOUND_ANSWER, [], [], None)
    verdict = {"id": 1, "missing_points": [], "wrong_points": [], "faithful": True, "reason": "đủ"}

    records = apply_verdicts([in_scope, out_scope], [verdict])

    assert records[0]["correct"] and records[0]["faithful"]
    assert records[1]["correct"] and records[1]["faithful"] is None


def test_apply_verdicts_fails_on_missing_id():
    record = score(A_STAR, "f = g + h [1]", ["03.txt > 5. A*"], ["03.txt"], None)

    with pytest.raises(ValueError, match=r"\[1\]"):
        apply_verdicts([record], [])


def test_score_records_missing_points():
    lenient = Verdict(missing_points=["α, β"], wrong_points=[], faithful=True, reason="thiếu")

    record = score(A_STAR, "f = g + h [1]", ["03.txt > 5. A*"], ["03.txt"], lenient)

    assert not record["correct"]
    assert record["missing_points"] == ["α, β"]


def test_expected_files_accepts_none_str_and_list():
    assert expected_files(OUT) == []
    assert expected_files(A_STAR) == ["03.txt"]
    assert expected_files(MULTI) == ["06.txt", "07.txt"]


def test_all_question_files_have_required_fields():
    questions = load_questions()

    assert len(questions) == 40
    assert len({q["id"] for q in questions}) == 40
    for q in questions:
        assert {"id", "question", "expected_answer", "source_file", "section"} <= q.keys()


def test_test_split_is_separate_from_dev():
    dev, test = load_questions("dev"), load_questions("test")

    assert len(test) == 30
    assert {q["id"] for q in dev}.isdisjoint(q["id"] for q in test)
    assert {q["question"] for q in dev}.isdisjoint(q["question"] for q in test)
    for q in test:
        assert {"id", "question", "expected_answer", "source_file", "section"} <= q.keys()


def test_score_correct_in_scope_answer():
    record = score(A_STAR, "f = g + h [1]", ["03.txt > 5. A*"], ["03.txt", "02.txt"], GOOD)

    assert record["correct"] and record["source_ok"] and record["retrieval_hit"] and record["faithful"]


def test_score_false_refusal_is_wrong():
    record = score(A_STAR, NOT_FOUND_ANSWER, [], ["03.txt"], None)

    assert record["refused"] and not record["correct"] and not record["source_ok"]


def test_score_multi_lecture_needs_every_file():
    record = score(MULTI, "x [1]", ["06.txt > 5. MINIMAX"], ["06.txt", "02.txt"], GOOD)

    assert record["correct"]
    assert not record["source_ok"]
    assert not record["retrieval_hit"]


def test_score_out_of_scope():
    refused = score(OUT, f"{NOT_FOUND_ANSWER}.", [], ["03.txt"], None)
    answered = score(OUT, "SVM tìm siêu phẳng [1]", ["03.txt > 5. A*"], ["03.txt"], None)

    assert refused["correct"] and refused["source_ok"] and refused["retrieval_hit"] is None
    assert not answered["correct"] and not answered["source_ok"]


def test_summarize_reports_accuracy_and_wrong_ids():
    records = [
        score(A_STAR, "f = g + h [1]", ["03.txt > 5. A*"], ["03.txt"], GOOD),
        score(OUT, "SVM tìm siêu phẳng [1]", ["03.txt > 5. A*"], ["03.txt"], None),
    ]

    summary = summarize(records)

    assert "Độ chính xác:          1/2 (50%)" in summary
    assert "Câu sai (cần duyệt tay): [3]" in summary
