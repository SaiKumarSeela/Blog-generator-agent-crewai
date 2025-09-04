import json
import pytest

from src.blog_generator_ai_agent.crew import BlogGeneratorCrew


@pytest.fixture
def crew(monkeypatch):
    # Avoid hitting external services within LLM initialization by setting dummy key
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    return BlogGeneratorCrew()


def test_extract_crew_output_handles_plain_string_json(crew):
    raw = '{"topic": "x", "blog_content": "hello world"}'

    class Obj:
        def __init__(self, s):
            self.raw = s

    parsed = crew._extract_crew_output(Obj(raw))
    assert isinstance(parsed, dict)
    assert parsed["blog_content"] == "hello world"
    assert "metadata" in parsed and "word_count" in parsed["metadata"]


def test_extract_crew_output_handles_markdown_fenced_json(crew):
    raw = """```json
    {"topic": "t", "blog_content": "alpha beta"}
    ```"""

    class Obj:
        def __init__(self, s):
            self.result = s

    parsed = crew._extract_crew_output(Obj(raw))
    assert parsed.get("topic") == "t"
    assert parsed.get("metadata", {}).get("word_count", 0) >= 2


