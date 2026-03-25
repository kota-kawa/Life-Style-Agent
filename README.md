> 一番下に日本語版もあります。

# Life-Style-Agent

<img src="static/Life-Style-Agent-Logo.png" width="800" alt="Life-Style-Agent logo">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-00629B?logo=meta&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?logo=huggingface&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-Server-6B21A8?logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-API-8E75B2?logo=google&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-API-D97757?logo=anthropic&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-API-F55036?logo=groq&logoColor=white)

## UI Preview

<p align="center">
  <img src="assets/images/Life-Syle-Agent-Screenshot.png" width="1000" alt="Life-Style-Agent web UI screenshot">
</p>

## 🎬 Demo Videos

Click a thumbnail to open the video on YouTube.

| [![Demo Video 1](https://img.youtube.com/vi/Ekhm9XJBhUg/hqdefault.jpg)](https://youtu.be/Ekhm9XJBhUg) | [![Demo Video 2](https://img.youtube.com/vi/Qoidfk93CHk/hqdefault.jpg)](https://youtu.be/Qoidfk93CHk) |
| --- | --- |
| Looking up tonight's dinner menu | What to watch out for in NISA (stock) investing |

Life-Style-Agent is an intelligent assistant with a RAG (Retrieval-Augmented Generation) engine for answering everyday life questions across domains like career, finance, cooking, home appliances, lifestyle, mental health, and society.

## Features

- **RAG (Retrieval-Augmented Generation):** Generates grounded answers using a FAISS vector database.
- **Multi-domain knowledge:** Covers careers, finance, cooking, home appliances, lifestyle, mental health, and society.
- **MCP (Model Context Protocol) support:** Runs as an MCP server for external agents/tools.
- **Web interface:** Simple chat UI for end users.
- **Conversation analysis:** Detects when support might be needed based on logs.
- **Multi-model support:** Gemini, OpenAI, Anthropic, Groq, and more.

## Tech Stack

- **Backend:** FastAPI (Python)
- **Vector DB:** FAISS
- **Embeddings:** HuggingFace (`intfloat/multilingual-e5-large`)
- **Frameworks:** LangChain, LlamaIndex
- **Frontend:** HTML/CSS/JS (Vanilla)
- **Infrastructure:** Docker, Docker Compose

## Directory Structure

```text
.
├── app.py                # FastAPI web app entrypoint
├── mcp_server.py         # MCP server implementation
├── lifestyle_agent/      # Core logic
│   ├── core/             # RAG engine (FAISS)
│   ├── api/              # MCP tool definitions
│   └── config/           # Paths, models, env config
├── data/                 # Data directory
│   ├── qa_jsonl/         # Source QA data
│   ├── vdb/faiss/        # Persisted FAISS indexes
│   └── home-topic/       # Raw documents (docx, etc.)
├── scripts/ingestion/    # Data ingestion & indexing scripts
├── web/                  # Web UI (templates & static files)
└── docs/                 # Docs & setup notes
```

## Quick Start (Docker Compose)

### 1) Prerequisites

- Docker
- Docker Compose (v2)

### 2) Configure secrets

Create `secrets.env` and set at least one API key.

```env
GOOGLE_API_KEY=your_api_key_here
# or
OPENAI_API_KEY=your_api_key_here
```

### 3) Start the app

```bash
docker compose up --build
```

Open `http://localhost:5000` in your browser.

### 4) (Optional) Build FAISS indexes

If you update the source corpus, rebuild the indexes inside the container:

```bash
docker compose run --rm qasystem python scripts/ingestion/jsonl_to_vector_faiss.py
```

### 5) Stop the app

```bash
docker compose down
```

## MCP Server Usage

The MCP server is available via `mcp_server.py` and exposes the `rag_answer` tool for MCP-compatible clients.

## Evaluation

### Life-Style Agent

**Role**
The Life-Style Agent is a RAG-based knowledge agent specialized for everyday support, such as cooking, finance, mental wellness, and home knowledge.

**Evaluation Protocol**
I prepared domain-specific QA tasks and evaluated whether the generated answers included the required core concepts from retrieved knowledge.

**Result**
The evaluation showed that the agent could consistently ground its answers in domain-relevant information across multiple lifestyle domains.

**Interpretation**
For this agent, the key research point is not only "which LLM is stronger," but also **how retrieval quality and grounding design affect downstream answer reliability**.

**Why this matters**
This section demonstrates that I did not treat RAG as a black box; I evaluated it as a retrieval-conditioned reasoning system.

## Development Guidelines

See [AGENTS.md](AGENTS.md) for coding style, testing guidance, and operational notes.

---

<details>
<summary>日本語</summary>

# Life-Style-Agent

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-00629B?logo=meta&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?logo=huggingface&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-Server-6B21A8?logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-API-8E75B2?logo=google&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-API-D97757?logo=anthropic&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-API-F55036?logo=groq&logoColor=white)

## UIプレビュー

<p align="center">
  <img src="assets/images/Life-Syle-Agent-Screenshot.png" width="1000" alt="Life-Style-Agent web UI screenshot">
</p>

## 🎬 デモ動画

サムネイルをクリックすると、YouTubeで動画を開けます。

| [![デモ動画1](https://img.youtube.com/vi/Ekhm9XJBhUg/hqdefault.jpg)](https://youtu.be/Ekhm9XJBhUg) | [![デモ動画2](https://img.youtube.com/vi/Qoidfk93CHk/hqdefault.jpg)](https://youtu.be/Qoidfk93CHk) |
| --- | --- |
| 今日の晩御飯の献立を調べる | NISA(株式）投資で気を付けることを調べる |

Life-Style-Agentは、家庭生活における多様なトピックに関する質問に答えるための、RAG（Retrieval-Augmented Generation）エンジンを備えた知的エージェントです。

## 特徴

- **RAG（Retrieval-Augmented Generation）:** FAISSベクトルデータベースを活用し、信頼性の高い回答を生成します。
- **マルチドメイン対応:** キャリア、金融、料理、家電、ライフスタイル、メンタルヘルス、社会などに対応。
- **MCP（Model Context Protocol）対応:** 他のエージェントやツールから呼び出し可能。
- **Webインターフェース:** 直感的なチャットUI。
- **会話分析:** 会話ログからサポートが必要なタイミングを検知。
- **マルチモデル対応:** Gemini、OpenAI、Anthropic、Groqなどに対応。

## 技術スタック

- **Backend:** FastAPI (Python)
- **Vector DB:** FAISS
- **Embeddings:** HuggingFace (`intfloat/multilingual-e5-large`)
- **Frameworks:** LangChain, LlamaIndex
- **Frontend:** HTML/CSS/JS (Vanilla)
- **Infrastructure:** Docker, Docker Compose

## ディレクトリ構成

```text
.
├── app.py                # FastAPI Webアプリケーションのエントリポイント
├── mcp_server.py         # MCPサーバーの実装
├── lifestyle_agent/      # コアロジック
│   ├── core/             # RAGエンジン (FAISS)
│   ├── api/              # MCPツール定義
│   └── config/           # パス、モデル、環境変数設定
├── data/                 # データディレクトリ
│   ├── qa_jsonl/         # ソースとなるQAデータ
│   ├── vdb/faiss/        # 永続化されたFAISSインデックス
│   └── home-topic/       # 生ドキュメント (docx等)
├── scripts/ingestion/    # データ加工・インデックス作成スクリプト
├── web/                  # Web UI (テンプレート・静的ファイル)
└── docs/                 # ドキュメント・セットアップノート
```

## クイックスタート（Docker Compose）

### 1) 前提

- Docker
- Docker Compose（v2）

### 2) secrets.env の設定

`secrets.env` を作成し、いずれかのAPIキーを設定してください。

```env
GOOGLE_API_KEY=your_api_key_here
# または
OPENAI_API_KEY=your_api_key_here
```

### 3) 起動

```bash
docker compose up --build
```

起動後、`http://localhost:5000` にアクセスしてください。

### 4) （任意）FAISSインデックスの再構築

ソースのコーパスを更新した場合は、コンテナ内で再構築します。

```bash
docker compose run --rm qasystem python scripts/ingestion/jsonl_to_vector_faiss.py
```

### 5) 停止

```bash
docker compose down
```

## MCPサーバー利用

`mcp_server.py` を通じて、MCP対応クライアントから `rag_answer` ツールを利用できます。

## 評価

### ライフスタイルエージェント

**役割**
ライフスタイルエージェントは、料理・家計・メンタルヘルス・家事知識など、日常生活のサポートに特化したRAGベースの知識エージェントです。

**評価プロトコル**
ドメインごとのQAタスクを用意し、生成された回答が検索された知識から必要なコア概念を含んでいるかどうかを評価しました。

**結果**
評価の結果、エージェントは複数のライフスタイルドメインにわたって、ドメイン関連情報に一貫して根拠を置いた回答を生成できることが示されました。

**解釈**
このエージェントにとって重要な研究観点は「どのLLMが優れているか」だけでなく、**検索品質とグラウンディング設計が回答信頼性にどのような影響を与えるか**です。

**この評価の意義**
このセクションは、RAGをブラックボックスとして扱わず、検索条件付き推論システムとして評価したことを示しています。

## 開発ガイドライン

詳細は [AGENTS.md](AGENTS.md) を参照してください。

</details>
