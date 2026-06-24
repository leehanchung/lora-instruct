"""Phase 2: score rollouts offline.

Reads the rollout JSONL from phase 1 and scores each row with the SHARED
dr_agent reward registry — the exact scorers the RL reward uses. Cheap and
re-runnable: change a scorer, re-score without re-generating.
"""

from __future__ import annotations

import json
from pathlib import Path

from dr_agent.rewards import Row, score


def score_rollouts(rollouts_path: Path) -> dict:
    scores: list[float] = []
    per_row: list[dict] = []
    with rollouts_path.open() as f:
        for line in f:
            r = json.loads(line)
            result = score(
                Row(
                    data_source=r["data_source"],
                    prediction=r["prediction"],
                    ground_truth=r["ground_truth"],
                )
            )
            scores.append(result.score)
            per_row.append({"id": r["id"], "score": result.score})
    mean = sum(scores) / len(scores) if scores else 0.0
    return {"mean_score": mean, "n": len(scores), "rows": per_row}
