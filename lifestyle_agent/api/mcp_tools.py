"""Shared helpers for Life-Style API and MCP tools."""

from __future__ import annotations

from typing import Any, Dict

from lifestyle_agent.core import rag_engine_faiss as ai_engine


def run_rag_answer(question: str, *, persist_history: bool = True) -> Dict[str, Any]:
    """Invoke the RAG pipeline and return the API-shaped payload."""

    answer, sources = ai_engine.get_answer(question, persist_history=persist_history)
    return {"answer": answer, "sources": sources}

