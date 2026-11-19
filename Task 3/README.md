# Task 3 · Quantisation & Inference

Qwen2.5-1.5B-Instruct through Ollama on a Colab T4: Q4_K_M versus Q8_0, using the same 10 prompts with two repetitions per model.

[Report](reports/task_3_report.md) · [Notebook](notebook/Task_3.ipynb) · [Results](quantization_benchmark/)

| Folder | Contents |
| :--- | :--- |
| `notebook/` | Original Colab notebook, including saved outputs |
| `reports/` | Original assignment report |
| `quantization_benchmark/` | Detailed and summary benchmarks, prompt responses, validation, and configuration |

### Run in Google Colab

1. Upload `notebook/Task_3.ipynb` to [Google Colab](https://colab.research.google.com/) and select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
2. Run the cells in order to install and start Ollama, download both quantised models, verify GPU offloading, benchmark, and compare outputs.
3. Keep the original `/content/quantization_benchmark/` path. The notebook creates it in the Colab runtime; the repository contains downloaded results. Download generated CSV and JSON files before the runtime ends. No Google Drive mount is required.

Model downloads stay in the runtime and are not included in Git. Timings and GPU memory depend on the runtime used; the report records the original run.
