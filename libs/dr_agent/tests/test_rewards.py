"""Tests for the shared reward registry (used by both training and eval)."""

import pytest

from dr_agent.rewards import Row, score


def test_exact_match_hit():
    row = Row(data_source="exact_match", prediction="The Eiffel Tower", ground_truth="eiffel tower")
    assert score(row).score == 1.0


def test_exact_match_miss():
    row = Row(data_source="exact_match", prediction="Big Ben", ground_truth="Eiffel Tower")
    assert score(row).score == 0.0


def test_f1_partial_overlap():
    row = Row(data_source="f1", prediction="the quick brown fox", ground_truth="quick brown dog")
    s = score(row).score
    assert 0.0 < s < 1.0


def test_f1_accepts_list_ground_truth():
    row = Row(data_source="f1", prediction="paris", ground_truth=["london", "paris"])
    assert score(row).score == 1.0


def test_f1_empty_ground_truth_list_does_not_crash():
    # max() over an empty golds list must not raise; score is 0.0.
    row = Row(data_source="f1", prediction="anything", ground_truth=[])
    assert score(row).score == 0.0


def test_unknown_data_source_raises():
    with pytest.raises(KeyError):
        score(Row(data_source="nope", prediction="x", ground_truth="y"))
