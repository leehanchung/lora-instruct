"""Exact-match scorer (normalized)."""

from __future__ import annotations

import re
import string

from dr_agent.rewards.schemas import RewardResult, Row


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def score(row: Row) -> RewardResult:
    golds = row.ground_truth if isinstance(row.ground_truth, list) else [row.ground_truth]
    pred = _normalize(row.prediction)
    hit = any(_normalize(g) == pred for g in golds)
    return RewardResult(score=1.0 if hit else 0.0)
