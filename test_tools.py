import os
import json
import numpy as np
import pytest

from src.blog_generator_ai_agent.tools.custom_tool import (
    EnhancedRAGTool,
    ResearchModeTool,
    WebSearchTool,
)


class DummyEmbedder:
    def encode(self, texts):
        if isinstance(texts, list):
            return np.zeros((len(texts), 384), dtype=np.float32)
        return np.zeros((1, 384), dtype=np.float32)


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    # Ensure API keys are not set unless a test sets them explicitly
    monkeypatch.delenv("SERP_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)


@pytest.fixture
def patched_sentence_transformer(monkeypatch):
    # Patch SentenceTransformer used inside EnhancedRAGTool to avoid network/model load
    from src.blog_generator_ai_agent.tools import custom_tool as ct

    class _DummyST:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts):
            return DummyEmbedder().encode(texts)

    monkeypatch.setattr(ct, "SentenceTransformer", _DummyST)
    return _DummyST


def test_enhanced_rag_retrieve_no_documents_returns_error(patched_sentence_transformer):
    rag = EnhancedRAGTool(knowledge_base_path="knowledge_test/")
    # Ensure a clean state
    assert rag.documents == []

    result = rag.retrieve("any query", top_k=3)
    assert result["total_found"] == 0
    assert "error" in result
    assert result["error"].lower().startswith("no documents")


def test_web_search_tool_without_api_key_returns_error():
    tool = WebSearchTool()
    output = tool._run("test query", num_results=3)
    data = json.loads(output)
    assert "error" in data
    assert data["results"] == []


def test_research_mode_internal_knowledge_no_docs(monkeypatch, patched_sentence_transformer):
    # Patch EnhancedRAGTool inside ResearchModeTool to a fake with empty docs
    from src.blog_generator_ai_agent.tools import custom_tool as ct

    class FakeRag:
        def __init__(self, *args, **kwargs):
            self.documents = []

        def retrieve(self, topic, top_k):
            return {"retrieved_documents": [], "total_found": 0}

    monkeypatch.setattr(ct, "EnhancedRAGTool", FakeRag)

    research_tool = ResearchModeTool()
    data = research_tool.internal_knowledge_retrieval("unit testing", top_k=5)
    assert data["research_mode"] == "internal_knowledge_base"
    
    assert "findings" in data and data["findings"] == []


def test_research_mode_serp_without_api_key_returns_error(monkeypatch, patched_sentence_transformer):
    research_tool = ResearchModeTool()
    data = research_tool.serp_analysis("pytest mocking", num_results=2)
    assert data["findings"] == []


