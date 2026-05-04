import pytest

from utils.prompter import Prompter


@pytest.fixture
def prompter(project_cwd):
    return Prompter("alpaca")


def test_default_template_loads(project_cwd):
    p = Prompter()
    assert p.template["description"]
    assert "{instruction}" in p.template["prompt_no_input"]
    assert "{instruction}" in p.template["prompt_input"]
    assert "{input}" in p.template["prompt_input"]


def test_unknown_template_raises(project_cwd):
    with pytest.raises(ValueError, match="Can't read"):
        Prompter("does-not-exist")


def test_prompt_with_input(prompter):
    out = prompter.generate_prompt("translate to French", "hello")
    assert "translate to French" in out
    assert "hello" in out
    assert out.endswith("### Response:\n")


def test_prompt_without_input(prompter):
    out = prompter.generate_prompt("write a haiku")
    assert "write a haiku" in out
    assert "### Input:" not in out
    assert out.endswith("### Response:\n")


def test_prompt_with_label_appends(prompter):
    out = prompter.generate_prompt("say hi", label="hi there")
    assert out.endswith("hi there")


def test_get_response_extracts_after_marker(prompter):
    full = prompter.generate_prompt("say hi") + "hi there"
    assert prompter.get_response(full) == "hi there"


def test_empty_template_name_uses_alpaca_default(project_cwd):
    default_p = Prompter()
    explicit_p = Prompter("alpaca")
    assert default_p.template == explicit_p.template
