"""Token-level F1 word-overlap scorer (à la DeepResearcher format_and_f1)."""

from __future__ import annotations

from collections import Counter

from dr_agent.rewards.em import _normalize
from dr_agent.rewards.schemas import RewardResult, Row


def _f1(pred: str, gold: str) -> float:
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def score(row: Row) -> RewardResult:
    golds = row.ground_truth if isinstance(row.ground_truth, list) else [row.ground_truth]
    best = max((_f1(row.prediction, g) for g in golds), default=0.0)
    return RewardResult(score=best, components={"f1": best})
