"""Test phần tính độ đo retrieval (không mở index)."""
from eval.retrieval_hits import expected_pairs, is_hit, rank_of, reciprocal_rank

SINGLE = {"source_file": "03.txt", "section": "5. A*"}
MULTI = {"source_file": ["06.txt", "07.txt"], "section": ["5. MINIMAX", "2. WORST-CASE"]}


def test_expected_pairs_single_and_multi():
    assert expected_pairs(SINGLE) == [("03.txt", "5. A*")]
    assert expected_pairs(MULTI) == [("06.txt", "5. MINIMAX"), ("07.txt", "2. WORST-CASE")]


def test_rank_of_returns_first_position_or_none():
    assert rank_of("b", ["a", "b", "b"]) == 2
    assert rank_of("x", ["a"]) is None


def test_is_hit_needs_every_target_within_k():
    assert is_hit([1, 3], k=4)
    assert not is_hit([1, 5], k=4)
    assert not is_hit([1, None], k=10)


def test_reciprocal_rank_uses_best_rank():
    assert reciprocal_rank([None, 2]) == 0.5
    assert reciprocal_rank([None]) == 0.0
