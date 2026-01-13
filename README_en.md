# Life-Style-Agent

[日本語](README.md) | [English](README_en.md)

<img src="static/Life-Style-Agent-Logo.png" width="800">

Life-Style-Agent is an intelligent agent equipped with a RAG (Retrieval-Augmented Generation) engine designed to answer questions regarding various topics in home life.

It integrates expertise in career, finance, cooking, home appliances, lifestyle, mental health, and social aspects to support the user's daily life.

## 🌟 Features

- **RAG (Retrieval-Augmented Generation):** Utilizes FAISS vector database to generate answers based on reliable documents.
- **Multi-Domain Support:** Covers multiple specialized fields (Career, Facility, Finance, General Cooking, Home Appliances, Lifestyle Optimization, Mental Health, Social).
- **MCP (Model Context Protocol) Support:** Functions as an MCP server callable by other agents and tools.
- **Web Interface:** Features an intuitive chat UI.
- **Conversation Analysis:** Analyzes external conversation logs to automatically detect when support is needed.
- **Multi-Model Support:** Supports latest LLMs such as Gemini, OpenAI, Anthropic, and Groq.

## 🛠 Tech Stack

- **Backend:** FastAPI (Python)
- **Vector DB:** FAISS
- **Embeddings:** HuggingFace (`intfloat/multilingual-e5-large`)
- **Frameworks:** LangChain, LlamaIndex
- **Frontend:** HTML/CSS/JS (Vanilla)
- **Infrastructure:** Docker, Docker Compose

## 📁 Directory Structure

```text
.
├── app.py                # Entry point for FastAPI Web Application
├── mcp_server.py         # Implementation of MCP Server
├── lifestyle_agent/      # Core Logic
│   ├── core/             # RAG Engine (FAISS)
│   ├── api/              # MCP Tool Definitions
│   └── config/           # Path, Model, and Environment Variable Settings
├── data/                 # Data Directory
│   ├── qa_jsonl/         # Source QA Data
│   ├── vdb/faiss/        # Persisted FAISS Indices
│   └── home-topic/       # Raw Documents (docx, etc.)
├── scripts/ingestion/    # Data Processing and Index Creation Scripts
├── web/                  # Web UI (Templates and Static Files)
└── docs/                 # Documentation and Setup Notes
```

## 🚀 Setup

### 1. Environment Setup

Python 3.12 or higher and [uv](https://docs.astral.sh/uv/) are required.

```bash
# Install dependencies (virtual environment is created automatically)
uv sync
```

### 2. Environment Variables

Create a `secrets.env` file and set the necessary API keys.

```env
GOOGLE_API_KEY=your_api_key_here
# or
OPENAI_API_KEY=your_api_key_here
```

### 3. Build Vector Database

Create FAISS indices from documents.

```bash
# Create FAISS from JSONL
python scripts/ingestion/jsonl_to_vector_faiss.py
```

## 💻 Usage

### Start Web Chat App

```bash
python app.py
```
After starting, access `http://localhost:5000` in your browser.

### Use as MCP Server

Via `mcp_server.py`, the following tools are available to MCP-compatible clients (like Cline):

- `rag_answer`: Answers questions related to daily life.

### Run with Docker

```bash
docker compose up --build
```

## 📝 Development Guidelines

Refer to [AGENTS.md](AGENTS.md) for details.
- **Coding Style:** PEP 8 compliant
- **Commit Messages:** Concise and clear messages (Japanese/English)
