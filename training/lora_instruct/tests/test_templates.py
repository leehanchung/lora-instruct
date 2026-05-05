import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"

REQUIRED_KEYS = {"description", "prompt_input", "prompt_no_input", "response_split"}


def test_alpaca_template_has_required_keys():
    template = json.loads((TEMPLATES_DIR / "alpaca.json").read_text())
    assert REQUIRED_KEYS.issubset(template.keys())


def test_alpaca_template_format_strings():
    template = json.loads((TEMPLATES_DIR / "alpaca.json").read_text())
    template["prompt_input"].format(instruction="x", input="y")
    template["prompt_no_input"].format(instruction="x")


def test_response_split_marker_appears_in_prompts():
    template = json.loads((TEMPLATES_DIR / "alpaca.json").read_text())
    marker = template["response_split"]
    assert marker in template["prompt_input"]
    assert marker in template["prompt_no_input"]


def test_every_template_file_is_valid():
    for path in TEMPLATES_DIR.glob("*.json"):
        template = json.loads(path.read_text())
        assert REQUIRED_KEYS.issubset(template.keys()), (
            f"{path.name} missing keys: {REQUIRED_KEYS - set(template.keys())}"
        )
