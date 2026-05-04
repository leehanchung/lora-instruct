import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

ALPACA_RECORD_KEYS = {"instruction", "input", "output"}


@pytest.mark.parametrize(
    "filename",
    [
        "alpaca_data.json",
        "alpaca_data_cleaned_archive.json",
        "alpaca_data_gpt4.json",
    ],
)
def test_alpaca_dataset_is_list_of_records(filename):
    path = DATASET_DIR / filename
    data = json.loads(path.read_text())
    assert isinstance(data, list)
    assert len(data) > 0
    sample = data[0]
    assert ALPACA_RECORD_KEYS.issubset(sample.keys()), (
        f"{filename}[0] missing keys: {ALPACA_RECORD_KEYS - set(sample.keys())}"
    )


def test_prompts_jsonl_is_jsonlines():
    path = DATASET_DIR / "prompts.jsonl"
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) > 0
    for line in lines:
        json.loads(line)
