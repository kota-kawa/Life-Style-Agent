import queue
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app as app_module


@pytest.fixture
def client():
    return TestClient(app_module.app)


def test_rag_answer_requires_question(client):
    res = client.post("/rag_answer", json={})
    assert res.status_code == 400
    assert res.json()["error"] == "質問を入力してください"


def test_rag_answer_success(client, monkeypatch):
    def fake_run_rag_answer(question, persist_history=True):
        assert question == "hello"
        assert persist_history is True
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(app_module, "run_rag_answer", fake_run_rag_answer)
    res = client.post("/rag_answer", json={"question": "hello"})
    assert res.status_code == 200
    assert res.json() == {"answer": "ok", "sources": []}


def test_agent_rag_answer_persist_false(client, monkeypatch):
    calls = {}

    def fake_run_rag_answer(question, persist_history=True):
        calls["question"] = question
        calls["persist_history"] = persist_history
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(app_module, "run_rag_answer", fake_run_rag_answer)
    res = client.post("/agent_rag_answer", json={"question": "agent"})
    assert res.status_code == 200
    assert res.json() == {"answer": "ok", "sources": []}
    assert calls == {"question": "agent", "persist_history": False}


def test_history_endpoints(client, monkeypatch):
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

    history_res = client.get("/conversation_history")
    assert history_res.status_code == 200
    assert history_res.json() == {"conversation_history": [{"role": "User", "message": "hi"}]}

    summary_res = client.get("/conversation_summary")
    assert summary_res.status_code == 200
    assert summary_res.json() == {"summary": "summary"}

    reset_res = client.post("/reset_history")
    assert reset_res.status_code == 200
    assert reset_res.json()["status"] == "Conversation history reset."
    assert reset_calls["count"] == 1


def test_model_settings_get_post(client, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "current_selection",
        lambda *_: {"provider": "openai", "model": "gpt-4o", "base_url": ""},
    )
    monkeypatch.setattr(app_module, "update_override", lambda *_: None)
    monkeypatch.setattr(app_module.ai_engine, "refresh_llm", lambda *_: None)

    get_res = client.get("/model_settings")
    assert get_res.status_code == 200
    assert get_res.json() == {
        "selection": {"provider": "openai", "model": "gpt-4o", "base_url": ""}
    }

    post_res = client.post(
        "/model_settings",
        json={"provider": "groq", "model": "test"},
        headers={"X-Platform-Propagation": "1"},
    )
    assert post_res.status_code == 200
    assert post_res.json()["status"] == "ok"


def test_analyze_conversation_validation_and_success(client, monkeypatch):
    empty_res = client.post("/analyze_conversation", json={"conversation_history": []})
    assert empty_res.status_code == 400
    assert empty_res.json()["error"] == "会話履歴が空です"

    monkeypatch.setattr(
        app_module,
        "analyze_conversation_payload",
        lambda *_: {"analyzed": True, "needs_help": False},
    )
    ok_res = client.post(
        "/analyze_conversation",
        json={"conversation_history": [{"role": "User", "message": "hi"}]},
    )
    assert ok_res.status_code == 200
    assert ok_res.json() == {"analyzed": True, "needs_help": False}


def test_mcp_messages_requires_session(client):
    res = client.post("/mcp/messages", json={"foo": "bar"})
    assert res.status_code == 404
    assert res.json()["error"] == "Session not found"


def test_mcp_messages_accepts(client):
    session_id = "test-session"
    q = queue.Queue()
    app_module._MCP_SESSIONS[session_id] = q
    try:
        res = client.post(f"/mcp/messages?session_id={session_id}", json={"ping": "pong"})
        assert res.status_code == 202
        assert res.json() == {"status": "accepted"}
        assert q.get(timeout=1) == {"ping": "pong"}
    finally:
        app_module._MCP_SESSIONS.pop(session_id, None)


def test_evaluation_data_and_tasks(client, monkeypatch, tmp_path):
    eval_file = tmp_path / "evaluation_data.json"
    monkeypatch.setattr(app_module.evaluation_manager, "EVAL_FILE", str(eval_file))

    data_res = client.get("/api/evaluation/data")
    assert data_res.status_code == 200
    assert data_res.json() == {"tasks": [], "results": []}

    add_res = client.post("/api/evaluation/tasks", json={"action": "add_samples"})
    assert add_res.status_code == 200
    assert add_res.json() == {"status": "added"}

    data_res = client.get("/api/evaluation/data")
    assert data_res.status_code == 200
    assert len(data_res.json()["tasks"]) > 0
