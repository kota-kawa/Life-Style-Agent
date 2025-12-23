from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
VDB_FAISS_DIR = DATA_DIR / "vdb" / "faiss"
HISTORY_FILE = DATA_DIR / "conversation_history.json"
EVALUATION_FILE = DATA_DIR / "evaluation" / "evaluation_data.json"
QA_JSONL_DIR = DATA_DIR / "qa_jsonl"
DOCX_SOURCE_DIR = DATA_DIR / "home-topic"

WEB_DIR = ROOT_DIR / "web"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

SECRETS_ENV = ROOT_DIR / "secrets.env"
INGESTION_SECRETS_ENV = DATA_DIR / "secrets" / "ingestion.env"
