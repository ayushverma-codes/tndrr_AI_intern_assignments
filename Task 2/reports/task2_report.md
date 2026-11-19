# Task 2 — RAG from Scratch

## 1. Objective and source

I built a retrieval-augmented question-answering system over the **NIST AI Risk Management Framework (AI RMF 1.0)**, a public **48-page PDF**. I implemented document processing and retrieval without LangChain or LlamaIndex, evaluated retrieval on 15 questions, investigated failures, and then added LLM inference. The work was carried out in Google Colab.

**Final flow:** PDF → page-aware chunks → MiniLM embeddings + BM25 → Reciprocal Rank Fusion (RRF) → evidence-relevance gate → focused passage selection → LLM answer → numerical-presence check. Source pages are retained with the evidence. I tested both local Qwen inference and an optional Groq API backend.

## 2. Implementation choices and trade-offs

| Component | What I used | Decision and trade-off |
|---|---|---|
| PDF ingestion | PyMuPDF; 48 pages | Extracts page text and preserves page numbers. Page-boundary chunking makes citations easier but can split cross-page context. |
| Chunking | **128 tokens / 24 overlap:** 212 chunks; **224 tokens / 32 overlap:** 126 chunks | Small chunks isolate definitions; large chunks preserve more context. Overlap helps at boundaries but repeats text. |
| Dense search | `all-MiniLM-L6-v2`, 384-dimensional normalized embeddings; NumPy dot products | Simple and reproducible for a small corpus, but semantic similarity does not guarantee direct answers. |
| Lexical search | BM25 coded in Python/NumPy | Rewards exact terminology, but can favor repeated terms over answer-bearing passages. |
| Hybrid ranking | RRF; 30 candidates per method, fusion constant 60 | Combines semantic and lexical rankings without comparing their incompatible scores. It can also introduce regressions. |
| Local inference | `Qwen/Qwen2.5-0.5B-Instruct` (~494M parameters) | Runs on Colab GPU with no paid API; less reliable at citation formatting and complete answers. |
| API inference | Groq `openai/gpt-oss-20b` | Produced more complete, cited answers in these tests; requires an API key, internet access, and rate-limit awareness. |

A focused evidence selector retrieves a larger shortlist, ranks short passage spans using MiniLM similarity and lexical overlap, then filters very short fragments. This was necessary because early sentence selection sometimes promoted meaningless fragments above definitions.

## 3. Retrieval evaluation

I wrote **15 questions** and annotated a supporting passage and PDF page for each. All 15 annotations were marked verified. A retrieved chunk counts as a match when it is on the annotated page and shares at least **12 consecutive normalized words** with the annotated passage. I measured Recall@1, Recall@3, Recall@5 and MRR@5. These are **annotated-evidence overlap metrics**, not answer-accuracy scores.

| Retrieval | Chunk size | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|---:|
| Dense | 128 | 66.67% | 80.00% | 93.33% | 0.7522 |
| Dense | 224 | 60.00% | 80.00% | 93.33% | 0.7078 |
| Hybrid (dense + BM25/RRF) | 128 | **73.33%** | **93.33%** | 93.33% | **0.8000** |
| Hybrid (dense + BM25/RRF) | 224 | 60.00% | **93.33%** | 93.33% | 0.7444 |

**Finding:** Hybrid search improved early ranking for both sizes, but not top-five coverage: **14/15** annotated passages were found in the top five in every configuration.

### Two inspected failures

**Q15 — Continuous AI risk management:** The annotated passage was ranked **14th** with 128-token dense chunks and **12th** with 224-token chunks. Hybrid retrieval moved it to **3rd** and **1st** respectively. The original dense results included some alternative relevant evidence, so this is specifically a *gold-passage ranking failure*.

**Q10 — Explainability versus interpretability:** The 224-token top-ranked dense chunk discussed benefits but did not directly define both concepts. The 128-token chunk contained the distinction. This is an *answer-completeness failure*, not simply a page-match failure. Section-aware chunking and cross-encoder re-ranking are possible improvements, but **I did not implement a cross-encoder**.

Hybrid search also introduced new misses: **Q08** with 128-token chunks and **Q03** with 224-token chunks. I retained these regressions in the evaluation.

## 4. LLM inference: what changed

I first tried extractive sentence selection, then local Qwen generation. The early Qwen pipeline returned answers to only **8/15** document questions because the other seven generated responses lacked the requested citation IDs and were rejected. Repeated prompting did not reliably fix citation formatting. A JSON answer-and-quote approach also failed on some cases: the model sometimes picked the wrong source or invented a quote.

I then gave Qwen **shorter, question-focused passages** and tracked evidence provenance in Python instead of relying on the model to format citations. This produced answers for **15/15** document questions. A preliminary qualitative review marked **10 answers adequate and five partial** (**Q05, Q06, Q07, Q09, Q14**). These labels were not an independent factual-accuracy assessment.

The focused generator initially hallucinated answers to **3/3 unrelated questions**. I added a lexical evidence gate (coverage threshold **0.45**) to abstain before inference when no passage matched enough important question terms. The gate alone still let through a harder question about an AI RMF financial penalty: Qwen invented a **$50,000 fine**. A subsequent **numerical-presence check** rejected that unsupported number. In the final Qwen regression run, it **answered 15/15 document questions and abstained on 7/7 tested unsupported questions**; the annotated gold page appeared in supplied evidence for all 15.

These guards are deliberately limited: sharing words with the source does not prove support, and finding a number somewhere in the evidence does not prove the associated claim. The pipeline does **not** provide independent claim-level citation verification.

## 5. Groq GPT-OSS-20B check

I also tested Groq's **`openai/gpt-oss-20b`** as an optional inference backend, using three focused passages per question. The first MEASURE test returned a correct answer citing page 33, using **407 tokens** with a **0.67-second** observed response time.

| Groq test | Observed result |
|---|---:|
| Document questions answered | **15/15** |
| Valid evidence-ID formats after normalizing `【E1】` to `[E1]` | **15/15** |
| Tokens across the 15-question run | **6,109** |
| Unsupported questions correctly rejected | **7/7** |
| Groq requests for the seven unsupported questions | **1** (428 tokens); other six stopped at the gate |

On the difficult financial-penalty question, Groq **abstained directly** rather than inventing a fine. The other six unsupported questions were rejected before API inference. Several Groq answers were more complete than Qwen's, particularly the trustworthy-AI characteristics and validity/reliability questions. Still, **Q05** was incomplete on risk prioritization and treatment, and **Q09** included a claim needing closer source verification. **15/15 answers and valid citation IDs do not mean 15/15 factually correct answers.**

This was an **exploratory comparison**, not a controlled model benchmark: Qwen used four focused passages in its final run, Groq used three, and their prompts differed. I kept Qwen as the API-free baseline and Groq as an optional backend.

## 6. BM25, re-ranking, and limitations

**BM25** scores exact-word matches using term frequency, term rarity and document-length normalization. It complements MiniLM's semantic matches. **RRF** combines the two retrieval rankings. This hybrid method was implemented and measured.

A **cross-encoder re-ranker** would score each question–passage pair jointly, potentially favoring a direct answer over a related paragraph, at added inference cost. I discussed this option but **did not implement or evaluate it**. The question-focused MiniLM evidence selector is not a cross-encoder.

Other limits: one 48-page source; 15 development questions rather than an untouched held-out test; one annotated passage per question even when alternatives exist; a small unsupported-question set; no statistical significance test; no independent factual-accuracy score; and no claim-level support verification. The saved artifact audit passed, but a fresh-runtime end-to-end rerun was not separately verified. API keys should be removed from the shared notebook.

## 7. Conclusion

I implemented PDF processing, two chunking configurations, embedding retrieval, BM25, hybrid rank fusion, evidence-based LLM generation, and abstention safeguards. **Hybrid retrieval improved early ranking without improving Recall@5**. Focused context improved local Qwen's answer coverage, while evidence gating and numerical checks caught the unsupported cases tested. Groq GPT-OSS-20B gave more consistently cited responses in an exploratory API test. The central lesson was that **retrieval quality, answer completeness, citation formatting, and factual support must be evaluated separately**.
