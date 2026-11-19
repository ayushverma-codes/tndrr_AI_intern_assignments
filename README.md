# tndrr · AI/ML Internship Assignment

My submission for the tndrr AI/ML internship assignment: three experiments covering model fine-tuning, retrieval, and inference optimisation. Each task includes its notebook, report, and saved results.

### 01 · Fine-tuning with LoRA

Fine-tuned **Qwen2.5-0.5B-Instruct** on Banking77 to classify customer queries into 77 banking intents. Compared the base and adapted models on 3,080 held-out examples, analysed failures, and estimated 7B training memory for full fine-tuning, LoRA, and QLoRA.

[Explore Task 1](Task%201/README.md) · [Read the report](Task%201/reports/task1_report.md)

### 02 · RAG from scratch

Built a question-answering system over the **48-page NIST AI Risk Management Framework**, using MiniLM embeddings, NumPy, and custom BM25 without LangChain or LlamaIndex. Compared two chunk sizes on 15 questions and added a Streamlit chatbot with source citations.

[Explore Task 2](Task%202/README.md) · [Read the report](Task%202/reports/task2_report.md)

### 03 · Model inference & optimisation

Ran **Qwen2.5-1.5B-Instruct** through Ollama on a Colab T4, comparing Q4_K_M and Q8_0. Measured generation speed and GPU memory, compared responses to the same 10 prompts, and discussed KV caching and a plan for serving 20 users on one 24 GB GPU.

[Explore Task 3](Task%203/README.md) · [Read the report](Task%203/reports/task_3_report.md)

---

**Running the work:** Tasks 1 and 3 use Google Colab with a GPU; follow their READMEs and keep the original notebook paths. Task 2 includes local setup instructions.

Model weights, checkpoints, dataset caches, secrets, and local environments are excluded from Git.
