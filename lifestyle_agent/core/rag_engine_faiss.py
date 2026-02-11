import os
import re
import json
import logging
from datetime import datetime
from typing import List, Optional

from lifestyle_agent.config.env import load_secrets_env
from lifestyle_agent.config.model_selection import apply_model_selection, update_override
from lifestyle_agent.config.paths import HISTORY_FILE, VDB_FAISS_DIR
from lifestyle_agent.core import prompt_guard

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

# ── 環境変数 ──
load_secrets_env()
logging.basicConfig(level=logging.DEBUG)


def refresh_llm(selection_override: dict | None = None):
    """Refresh the shared LLM instance based on selection."""
    global llm
    if selection_override:
        provider, model_name, _ = update_override(selection_override)
    else:
        provider, model_name, _ = apply_model_selection("lifestyle")

    provider = provider.lower()
    logging.info(f"Refreshing LLM: provider={provider}, model={model_name}")

    if provider == "gemini":
        llm = ChatGoogleGenerativeAI(model=model_name, convert_system_message_to_human=True)
    elif provider == "claude":
        llm = ChatAnthropic(model_name=model_name)
    elif provider == "groq":
        llm = ChatGroq(model_name=model_name)
    else:  # openai
        llm = ChatOpenAI(model=model_name)


refresh_llm()

# ── インデックス設定 ──
INDEX_DB_DIR = VDB_FAISS_DIR
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": EMBEDDING_DEVICE},
)


def _current_datetime_line() -> str:
    """Return the timestamp string embedded in prompts."""
    return datetime.now().strftime("現在の日時ー%Y年%m月%d日%H時%M分")


def _clean_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen model responses."""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()


def load_all_indices():
    """data/vdb/faiss 以下を走査し、FAISS のインデックスを読み込む"""
    if not os.path.exists(INDEX_DB_DIR):
        raise RuntimeError(f"Directory not found: {INDEX_DB_DIR}")

    combined_store: Optional[FAISS] = None
    
    # 再帰的に探索して index.faiss を探す
    for root, dirs, files in os.walk(INDEX_DB_DIR):
        if "index.faiss" in files:
            # root は index.faiss があるディレクトリ (通常は persist)
            persist_dir = root
            
            try:
                store = FAISS.load_local(
                    persist_dir,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                logging.info(f"Loaded FAISS index from {persist_dir}")
                
                if combined_store is None:
                    combined_store = store
                else:
                    combined_store.merge_from(store)
            except Exception:
                logging.exception(f"Failed to load FAISS index from {persist_dir}")
                continue

    if combined_store is None:
        raise RuntimeError("Failed to load any FAISS indices.")

    return combined_store


try:
    VECTOR_STORE = load_all_indices()
    vector_retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 5})
except Exception:
    logging.exception("FAISS インデックスの初期化に失敗しました。")
    vector_retriever = None


# ── 会話履歴ユーティリティ ──
def load_conversation_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("conversation_history", [])
        except Exception as e:
            logging.exception(f"履歴の読み込みエラー: {e}")
    return []


def save_conversation_history(hist):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"conversation_history": hist}, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.exception(f"履歴の保存エラー: {e}")


def get_conversation_history():
    """会話履歴を外部に提供するためのラッパー"""
    return load_conversation_history()


# ── プロンプト ──
COMBINE_PROMPT = """現在の日時ー{current_datetime}

あなたは、ライフスタイル全般をサポートする専門アシスタントです。
以下の専門知識を活かして、ユーザーの悩みや質問に具体的かつ実践的なアドバイスを提供します。

【重要：禁止事項】
- あなたはタスクの振り分けを行う「ルーター」や「オーケストレーター」ではありません。
- 「エージェント種別が指定されていません」「どのエージェントを使用しますか？」といった、エージェント選択に関する質問や確認は一切行わないでください。
- ユーザーの質問に対して、あなたの持つ知識と検索結果（RAG）を用いて直接回答してください。

【あなたの役割と専門分野】
1. **キャリア**: 転職、給与交渉、自己分析、職場の人間関係、スキルアップ
2. **金融・家計**: 資産運用（NISA等）、節約術、サブスク見直し、家計管理
3. **家電・設備**: 家電の選び方・トラブル対応、住宅設備のメンテナンス（排水溝、網戸など）
4. **メンタルヘルス**: ストレス解消、睡眠改善、モチベーション管理、不安の軽減
5. **料理・食事**: 時短レシピ、献立作成、自炊のコツ、食材管理
6. **生活最適化**: 引越し手続き、タイムマネジメント、習慣化、効率的な家事
7. **人間関係・ソーシャル**: ギフト選び、断り方、雑談のコツ、コミュニケーション術

【会話履歴】
{history}

【質問】
{question}

【回答候補（RAG検索結果）】
{summaries}

【回答ガイドライン】
- **具体的・実践的**: 抽象的なアドバイスではなく、「今すぐできること」や「具体的な手順」を提案してください。
- **資料の活用**: 提供された【回答候補】の情報を最優先で使用してください。
- **情報不足への対応**: 資料に情報がない場合は、その旨を伝えつつ、あなたの一般的知識（上記の専門分野に基づく）で補足してください。
- **共感と寄り添い**: ユーザーの状況に寄り添い、丁寧で温かみのあるトーンで回答してください。
- **根拠の明示**: 参照元がわかる場合は、自然な形で織り交ぜてください。
- **言語**: 必ず日本語で回答してください。
"""


# ── 公開 API ──
def _format_history_for_prompt(history: List[dict]) -> str:
    if not history:
        return "（履歴なし）"
    recent = history[-6:]
    return "\n".join(f"{entry['role']}: {entry['message']}" for entry in recent)


def get_answer(question: str, persist_history: bool = True):
    """質問文字列を受け取り、RAG 結果（answer, sources）を返す"""
    question = question.strip()
    if not question:
        raise ValueError("質問を入力してください。")

    history = load_conversation_history()

    guard_decision = prompt_guard.evaluate_prompt_guard(question)
    if guard_decision.block:
        final = prompt_guard.guard_refusal_message(guard_decision)
        if persist_history:
            updated_history = history + [
                {"role": "User", "message": question},
                {"role": "AI", "message": final},
            ]
            save_conversation_history(updated_history)
        return final, []

    if vector_retriever is None:
        raise RuntimeError("FAISS インデックスが初期化されていません。")

    retrieved_docs = vector_retriever.get_relevant_documents(question)

    context_blocks = []
    top_srcs = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "不明ファイル")
        if source not in top_srcs:
            top_srcs.append(source)
        header_parts = [f"候補{idx}: 出典={source}"]
        if doc.metadata.get("line") is not None:
            header_parts.append(f"行={doc.metadata['line']}")
        if doc.metadata.get("chunk_id") is not None:
            header_parts.append(f"チャンク={doc.metadata['chunk_id']}")
        header = " / ".join(header_parts)
        context_blocks.append(f"{header}\n{doc.page_content.strip()}")

    if context_blocks:
        summaries_text = "\n\n".join(context_blocks)
    else:
        summaries_text = "該当資料は見つかりませんでした。資料が不足する場合はその旨を伝えつつ、一般的な知識で補足してください。"

    prompt_text = COMBINE_PROMPT.format(
        current_datetime=_current_datetime_line(),
        history=_format_history_for_prompt(history),
        question=question,
        summaries=summaries_text,
    )

    raw_answer = llm.invoke(prompt_text)
    if isinstance(raw_answer, str):
        answer = raw_answer
    else:
        answer = getattr(raw_answer, "content", None) or getattr(raw_answer, "text", None) or str(raw_answer)
    answer = _clean_think_tags(answer)

    ref_dict = {s: set() for s in top_srcs[:3] if s and s != "不明ファイル"}
    for doc in retrieved_docs:
        src = doc.metadata.get("source")
        if src in ref_dict:
            page = doc.metadata.get("page") or doc.metadata.get("line") or doc.metadata.get("chunk_id")
            if page is not None:
                ref_dict[src].add(str(page))

    if ref_dict:
        refs = ", ".join(
            f"{s} (page: {', '.join(sorted(pgs))})" for s, pgs in ref_dict.items()
        )
        final = answer + "\n\n【使用したファイル】\n" + refs
    else:
        final = answer

    if persist_history:
        updated_history = history + [
            {"role": "User", "message": question},
            {"role": "AI", "message": final},
        ]
        save_conversation_history(updated_history)

    return final, list(ref_dict.keys())


def reset_history():
    """data/conversation_history.json を空にする"""
    save_conversation_history([])


SUMMARY_PROMPT = """あなたは会話を俯瞰し、重要なポイントを簡潔にまとめるアシスタントです。

【会話履歴】
{history}

【要約ルール】
- 重要な質問や回答、決定事項、未解決の点などを中心に2〜5行でまとめてください。
- 会話がない場合は、その旨を1行で伝えてください。
- 必ず日本語で出力してください。
"""


def _format_history_for_summary(history: List[dict], limit: int = 20) -> str:
    if not history:
        return "（会話はありません）"
    recent = history[-limit:]
    return "\n".join(f"{entry['role']}: {entry['message']}" for entry in recent)


def summarize_conversation(history: List[dict]) -> str:
    if not history:
        return "まだ会話はありません。"

    formatted_history = _format_history_for_summary(history)
    prompt_text = SUMMARY_PROMPT.format(history=formatted_history)

    summary_response = llm.invoke(prompt_text)
    if isinstance(summary_response, str):
        return _clean_think_tags(summary_response)
    result = (
        getattr(summary_response, "content", None)
        or getattr(summary_response, "text", None)
        or str(summary_response)
    )
    return _clean_think_tags(result)


def get_conversation_summary() -> str:
    history = load_conversation_history()
    return summarize_conversation(history)
