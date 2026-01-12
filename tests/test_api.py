import asyncio
import queue
import sys
from pathlib import Path
import json

import pytest
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app as app_module


class _FakeRequest:
    def __init__(self, payload=None, query_params=None, headers=None, method="POST"):
        self._payload = payload
        self.query_params = query_params or {}
        self.headers = headers or {}
        self.method = method

    async def json(self):
        return self._payload


def _json_from_response(response):
    if isinstance(response, JSONResponse):
        return json.loads(response.body.decode("utf-8"))
    return response


@pytest.fixture(autouse=True)
def _inline_thread_calls(monkeypatch):
    import anyio

    async def _run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(anyio.to_thread, "run_sync", _run_sync)


def test_rag_answer_requires_question():
    res = asyncio.run(app_module.rag_answer(_FakeRequest({})))
    payload = _json_from_response(res)
    assert payload["error"] == "質問を入力してください"


def test_rag_answer_success(monkeypatch):
    def fake_run_rag_answer(question, persist_history=True):
        assert question == "hello"
        assert persist_history is True
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(app_module, "run_rag_answer", fake_run_rag_answer)
    res = asyncio.run(app_module.rag_answer(_FakeRequest({"question": "hello"})))
    assert res == {"answer": "ok", "sources": []}


def test_agent_rag_answer_persist_false(monkeypatch):
    calls = {}

    def fake_run_rag_answer(question, persist_history=True):
        calls["question"] = question
        calls["persist_history"] = persist_history
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(app_module, "run_rag_answer", fake_run_rag_answer)
    res = asyncio.run(app_module.agent_rag_answer(_FakeRequest({"question": "agent"})))
    assert res == {"answer": "ok", "sources": []}
    assert calls == {"question": "agent", "persist_history": False}


def test_history_endpoints(monkeypatch):
    monkeypatch.setattr(
        app_module.ai_engine,
        "get_conversation_history",
        lambda: [{"role": "User", "message": "hi"}],
    )
    monkeypatch.setattr(app_module.ai_engine, "get_conversation_summary", lambda: "summary")
    reset_calls = {"count": 0}

    def fake_reset():
        reset_calls["count"] += 1

    monkeypatch.setattr(app_module.ai_engine, "reset_history", fake_reset)

    history_res = asyncio.run(app_module.conversation_history())
    assert history_res == {"conversation_history": [{"role": "User", "message": "hi"}]}

    summary_res = asyncio.run(app_module.conversation_summary())
    assert summary_res == {"summary": "summary"}

    reset_res = asyncio.run(app_module.reset_history())
    assert reset_res["status"] == "Conversation history reset."
    assert reset_calls["count"] == 1


def test_analyze_conversation_endpoint_removed():
    # Function removed; FastAPI would return 404 when mounted.
    assert not hasattr(app_module, "analyze_conversation")


def test_model_settings_get_post(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "current_selection",
        lambda *_: {"provider": "openai", "model": "gpt-4o", "base_url": ""},
    )
    monkeypatch.setattr(app_module, "update_override", lambda *_: None)
    monkeypatch.setattr(app_module.ai_engine, "refresh_llm", lambda *_: None)

    get_res = asyncio.run(app_module.update_model_settings(_FakeRequest(method="GET")))
    assert get_res == {"selection": {"provider": "openai", "model": "gpt-4o", "base_url": ""}}

    post_res = asyncio.run(
        app_module.update_model_settings(
            _FakeRequest({"provider": "groq", "model": "test"}, headers={"X-Platform-Propagation": "1"})
        )
    )
    assert post_res["status"] == "ok"


def test_mcp_messages_requires_session():
    res = asyncio.run(app_module.mcp_messages_endpoint(_FakeRequest({}, query_params={})))
    payload = _json_from_response(res)
    assert payload["error"] == "Session not found"


def test_mcp_messages_accepts():
    session_id = "test-session"
    q = queue.Queue()
    app_module._MCP_SESSIONS[session_id] = q
    try:
        res = asyncio.run(app_module.mcp_messages_endpoint(_FakeRequest({"ping": "pong"}, query_params={"session_id": session_id})))
        payload = _json_from_response(res)
        assert payload == {"status": "accepted"}
        assert q.get(timeout=1) == {"ping": "pong"}
    finally:
        app_module._MCP_SESSIONS.pop(session_id, None)


def test_evaluation_data_and_tasks(monkeypatch, tmp_path):
    eval_file = tmp_path / "evaluation_data.json"
    monkeypatch.setattr(app_module.evaluation_manager, "EVAL_FILE", str(eval_file))

    data_res = asyncio.run(app_module.get_eval_data())
    assert data_res == {"tasks": [], "results": []}

    add_res = asyncio.run(app_module.manage_eval_tasks(_FakeRequest({"action": "add_samples"}, method="POST")))
    assert add_res == {"status": "added"}

    data_res = asyncio.run(app_module.get_eval_data())
    assert len(data_res["tasks"]) > 0
