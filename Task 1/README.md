# Task 1 · LoRA Fine-tuning

Qwen2.5-0.5B-Instruct adapted to the 77 Banking77 intents, evaluated against the base model on 3,080 held-out examples.

[Report](reports/task1_report.md) · [Notebook](notebook/Banking77_Qwen_LoRA_Finetuning_Final_v2.ipynb) · [Results](Banking77_LoRA/results/)

| Folder | Contents |
| :--- | :--- |
| `notebook/` | Original Colab notebook, including saved outputs |
| `reports/` | Original assignment report |
| `Banking77_LoRA/results/` | Metrics, predictions, failure examples, and confusion matrices |
| `Banking77_LoRA/lora_training/` | Training history, metrics, and loss plot; checkpoints ignored |
| `Banking77_LoRA/lora_adapter/` | Adapter and experiment configurations; weights and tokenizer copies ignored |

### Run in Google Colab

1. Upload the notebook to [Google Colab](https://colab.research.google.com/) and select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
2. Run the cells in order, including dependency installation, and authorize the Google Drive mount. The notebook downloads the model and dataset, creates the dataset splits, trains LoRA, and evaluates both models.
3. Keep its original paths: outputs are written under `/content/drive/MyDrive/Banking77_LoRA/` in `dataset/`, `results/`, `lora_training/`, and `lora_adapter/`. The repository folders contain downloaded copies; they do not replace the Drive paths.

Training recreates the Drive `lora_training/` and `lora_adapter/` folders, so use a Drive account with no run you need to preserve at those paths. A full rerun regenerates the dataset and adapter; ignored local files are not needed for a fresh run. Duplicate exports remain locally in `local_exports/`.
