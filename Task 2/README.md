# NIST AI RMF · RAG Chatbot

**From notebook experiments to a modular Python application.**

Ask questions about the 48-page NIST AI Risk Management Framework through a Streamlit chatbot, with retrieved evidence and PDF page citations.

## The problem

Build a question-answering system over a public document of at least 30 pages, without LangChain or LlamaIndex. Compare two chunk sizes, evaluate retrieval on 15 annotated questions, and investigate retrieval failures.

## Starting point: the notebook

The project began in [`notebook/RAG_From_Scratch_NIST.ipynb`](notebook/RAG_From_Scratch_NIST.ipynb). It contains the original document processing, chunk-size experiments, custom BM25 and hybrid retrieval, generation experiments, and evaluation.

The application refactors that work into reusable modules and adds a chatbot. The original notebook and research files are preserved unchanged. See the [assignment report](reports/task2_report.md) for findings and trade-offs.

## How it works

```text
NIST PDF → Token chunks → MiniLM embeddings + BM25 → RRF hybrid retrieval
         → Relevance gate → Focused evidence → Groq → Citation & number checks
```

- **Index:** 212 page-aware chunks, 128 tokens with 24-token overlap.
- **Retrieval:** normalized MiniLM embeddings, NumPy similarity, custom BM25, and Reciprocal Rank Fusion.
- **Answers:** Groq `openai/gpt-oss-20b`, using three focused evidence passages.
- **Interface:** session chat history, PDF page references, and expandable evidence.

The selected configuration achieved **Recall@5: 0.9333** and **MRR@5: 0.8000** in the original 15-question experiment. These measure retrieval evidence coverage, not answer accuracy.

## Project structure

```text
.
├── app.py                 # Streamlit chatbot
├── config.py              # Settings and environment loading
├── ingestion.py           # PDF, token chunks, and embeddings
├── retrieval.py           # BM25, RRF, gating, and evidence selection
├── generation.py          # Groq inference and output safeguards
├── pipeline.py            # Reusable RAGPipeline.ask() interface
├── notebook/              # Original experimental notebook
├── data/                  # NIST source PDF
├── artifacts/             # Saved index and research artifacts
├── evaluation/            # Annotated questions and experiment results
├── reports/               # Original assignment report
├── tests/                 # Unit, regression, and integration tests
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
└── README.md
```

## Run locally

From the project root in **PowerShell**:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Skip this copy if you already have a .env file.
Copy-Item .env.example .env
```

Set `GROQ_API_KEY=your_key` in `.env`, then start the chatbot:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

MiniLM downloads on first use if it is not cached. Saved index artifacts are reused; missing artifacts can be rebuilt. Secrets, environments, model weights, and intermediate experiment files are excluded from Git.

**Try:** “What is the purpose of the MEASURE function?” or “What are the four core functions of the AI RMF?”

## Tests

```powershell
# Unit, retrieval regression, and Streamlit application checks
.\.venv\Scripts\python.exe -m pytest -q -m "not live"

# Real Groq integration tests — requires a valid key and network access
$env:RUN_GROQ_TESTS='1'
.\.venv\Scripts\python.exe -m pytest -q -m live
```

Latest root-level validation: **21 tests passed**, including four real Groq requests. All 15 historical retrieval rankings and focused-evidence selections matched the saved experiments.

## Limitations

Citation checks validate evidence identifiers; numerical checks look for numbers in the supplied text. Neither guarantees factual support. The relevance gate can accept unrelated questions with overlapping words, and the model can misinterpret source text—for example, answering an identity question as if it were a document-defined AI actor. This is a single-document research demonstration, with no independent claim-level verification.
