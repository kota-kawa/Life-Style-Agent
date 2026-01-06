# セットアップ/運用メモ

## 仮想環境
Python 3.12 以上を推奨します。
```
python3.12 -m venv venv
source venv/bin/activate
deactivate
```

## クリーニング
```
find . -type f -name '*.Identifier' -delete
```

## 依存関係（参考）
```
pip install PyPDF2 python-docx llama-index==0.12.30 langchain==0.3.23 langchain_google_genai==2.1.2 \
  langchain-community==0.3.21 llama-index-embeddings-huggingface==0.5.3 llama-index-embeddings-google-genai

pip install openai python-dotenv python-docx tenacity tqdm sentencepiece faiss-cpu
```

## 変換スクリプト

### 1) 単一 DOCX → JSONL（引数指定）
```
python scripts/ingestion/docx_to_qa_jsonl.py <input.docx> <output.jsonl>
```

### 2) ディレクトリ一括 DOCX → JSONL（thinking 固定）
```
DOCX_IN_DIR=./data/home-topic \
JSONL_OUT_DIR=./data/qa_jsonl \
python scripts/ingestion/docx_to_qa_jsonl_thinking.py
```

### 3) JSONL → FAISS
```
python scripts/ingestion/jsonl_to_vector_faiss.py
```

## secrets.env
- 基本はリポジトリ直下の `secrets.env` を使用
- 追加で `data/secrets/ingestion.env` を用意すると、変換スクリプトが優先的に読み込みます
