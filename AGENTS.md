# Repository Guidelines

## Architecture & Module Ownership
- `app.py` is the single Flask entrypoint. It wires CORS, loads environment secrets, and exposes the HTTP APIs consumed by the UI and other agents:
  - `/rag_answer` stores the exchanged turns and powers the user chat.
  - `/agent_rag_answer` serves peer agents without touching the persisted history.
  - `/reset_history`, `/conversation_history`, and `/conversation_summary` manage/chat through `data/conversation_history.json`.
  - `/analyze_conversation` inspects an external transcript and, when家庭科系の知識で解決できると判断した場合, turns it into a concrete VDB query plus support message.
  - `/` renders `web/templates/index.html`, whose inline JS handles the summary accordion, optimistic chat updates, loading spinners, and fetches listed above.
- `mcp_server.py` exposes `/mcp/sse` + `/mcp/messages` bridges for MCP clients and defines the `rag_answer` / `analyze_conversation` tools backed by the same logic as the HTTP APIs (install via `pip install "mcp[cli]"`).
- Retrieval and reasoning live in `lifestyle_agent/core/rag_engine_faiss.py`. It loads every `data/vdb/faiss/*/persist` FAISS store with `HuggingFaceEmbeddings` (`intfloat/multilingual-e5-large` on `EMBEDDING_DEVICE`), queries with `vector_retriever`, and calls the provider set by `lifestyle_agent/config/model_selection.py`.
  - `data/conversation_history.json` stores alternating `User`/`AI` turns; `reset_history()` clears it and the summary endpoint reuses the same file.
  - `get_conversation_summary()` and `analyze_external_conversation()` each run bespoke prompts so tweaks there must respect the JSON contracts in `app.py`.
- Shared path constants live in `lifestyle_agent/config/paths.py`. Update that file first when relocating data, static assets, or templates.
- Front-end assets live under `web/templates/` and `web/static/` (we currently only ship inline styles/JS). The UI expects JSON responses exactly as implemented in `app.py`; breaking shapes will surface immediately in fetch handlers.
- Data ingestion utilities live under `scripts/ingestion/`. Persist FAISS artifacts under `data/vdb/faiss/` and JSONL outputs under `data/qa_jsonl/`.
- Docker tooling (`Dockerfile`, `docker-compose.yml`) mirrors the local workflow: the `qasystem` service runs `flask run` with reload and attaches to the external `${MULTI_AGENT_NETWORK}` for agent-to-agent calls.

## Configuration & Secrets
- Put secrets in `secrets.env` (see `docs/setup_notes.md` for pointers). `GOOGLE_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY` must be defined; whichever exists first bootstraps both Gemini and LangChain's OpenAI client. Optional overrides: `OPENAI_BASE_URL`/`OPENAI_API_BASE`, `EMBEDDING_MODEL_NAME`, `EMBEDDING_DEVICE`.
- Vector jobs read from the same `secrets.env`, so ensure the keys are exported before running any ingestion scripts. Optional: `data/secrets/ingestion.env` can override for ingestion-only runs.

## Build & Run Commands
- Create a virtual environment with `python3.12 -m venv .venv && source .venv/bin/activate` (or reuse the checked-in `venv/` when allowed) and install dependencies via `pip install -r requirements.txt`.
- Local dev server: `FLASK_ENV=development python app.py` (or `flask run` equivalent). The Docker stack can be launched with `docker compose up --build qasystem`.
- Rebuild FAISS stores whenever the source corpus changes: run the relevant ingestion script (for example `python scripts/ingestion/docx_to_qa_jsonl.py` or `python scripts/ingestion/jsonl_to_vector_faiss.py`) and verify it writes only under the sanctioned vector folders.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation; prefer explicit imports and descriptive helpers (e.g., `get_conversation_summary`, `analyze_external_conversation`).
- Use snake_case for Python identifiers, kebab-case for CLI entry points, and keep HTML ids/classes consistent with the dash-delimited patterns already present in `web/templates/index.html`.
- Front-end changes should remain vanilla JS/CSS unless there is a strong reason otherwise; keep animations/lightweight DOM helpers readable and progressively enhanced.
- Run `black` on large edits and lint via `python -m compileall` until additional tooling lands.

## Manual & Automated Testing
- There is no automated suite yet. Add `pytest` cases under `tests/` using the `test_<module>.py` naming pattern when you touch Python core logic.
- Smoke tests to run after app or retrieval changes:
  - `curl -X POST localhost:5000/rag_answer -H 'Content-Type: application/json' -d '{"question":"..."}` should return `{answer,sources}` and append to `data/conversation_history.json`.
  - `curl -X POST localhost:5000/agent_rag_answer ...` must succeed without modifying history.
  - `curl localhost:5000/conversation_summary` verifies the summarizer output, and `curl -X POST localhost:5000/analyze_conversation` with a crafted transcript should either surface `"needs_help": true` plus `support_message` or cleanly decline.
- Update `evaluation/query_pipeline_stats.json` or any other benchmark artifacts whenever retrieval/ranking behavior changes.

## Commit & Pull Request Guidelines
- Keep commits short, present-tense (English or Japanese), and optionally scoped (`docs:`, `fix:`). Each commit should represent one logical change and note any vector rebuilds or regenerated assets in the body.
- PR descriptions must cover the motivation, manual/automated test steps (`python app.py`, ingestion script invocations, curl checks), and link related tickets or cooperating agents so reviewers understand downstream impact.

## Critical Instruction / 重要指示
どれだけ時間とコストをかけてもよいので、確実に目標が達成されるようにしてほしい。
