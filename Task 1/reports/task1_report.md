# Task 1: Fine-tuning with LoRA

## 1. Objective

The objective was to fine-tune an open-weight language model with fewer than 1.5 billion parameters using parameter-efficient fine-tuning. I compared the original model with the LoRA fine-tuned model on the same held-out test set. I also inspected prediction errors and estimated the GPU memory required to fine-tune a 7B model using full fine-tuning, LoRA and QLoRA.

## 2. Model and dataset

**Model:** `Qwen/Qwen2.5-0.5B-Instruct`  
**Parameters:** 494,032,768 (approximately 0.5B)  
**Dataset:** `PolyAI/banking77`

Banking77 contains customer-service queries grouped into 77 banking intent classes. The dataset originally provides 10,003 training examples and 3,080 test examples.

I split the original training set into 9,002 training examples and 1,001 validation examples using a stratified 90:10 split with seed 42. I kept the original 3,080-example test set untouched.

| Split | Examples | Use |
|---|---:|---|
| Train | 9,002 | LoRA training |
| Validation | 1,001 | Evaluation during training and best-checkpoint selection |
| Test | 3,080 | Final comparison |
| **Total** | **13,083** | |

All 77 classes were present in each split. The notebook also checked for text overlap between the splits and found none. The test set contains 40 examples per class.

## 3. Classification approach

I used the model as a text-generation classifier. Each prompt contains:
- A system instruction explaining the classification task.
- The list of all 77 allowed intent labels.
- The customer's query.
- An instruction to return only the exact intent label.

For baseline evaluation, I generated a prediction for every test query. I normalized the generated text and matched it against the allowed labels. Any output that could not be mapped to exactly one valid label was counted as invalid and treated as incorrect.

For LoRA training, I used the same prompt format. The expected intent label and the end-of-sequence token were appended as the target response. Prompt tokens were masked with `-100`, so the training loss was calculated on the assistant's target response rather than the prompt.

## 4. Base model evaluation

I loaded the base model in FP16 on a Google Colab Tesla T4 GPU. I evaluated it on all 3,080 test examples using a batch size of 4.

| Metric | Base model |
|---|---:|
| Accuracy | 18.93% |
| Macro F1 | 18.97% |
| Weighted F1 | 18.97% |
| Valid predictions | 2,264 |
| Invalid predictions | 816 |
| Invalid prediction rate | 26.49% |

The base model achieved low classification performance. A major issue was that it sometimes generated labels that were not part of the 77 allowed classes. These outputs could not be mapped to a valid intent and were counted as incorrect.

## 5. LoRA configuration and decisions

I used LoRA through the PEFT library. The base model weights were kept frozen, while small trainable adapter matrices were added to the attention projections.

| Setting | Value | Reason for selection |
|---|---:|---|
| LoRA rank (`r`) | 16 | Provides a moderate adapter capacity without training the full model. |
| LoRA alpha | 32 | Sets the scaling of the LoRA update; 32 gives an alpha-to-rank ratio of 2. |
| LoRA dropout | 0.05 | Adds a small amount of regularization to reduce overfitting. |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Adapts the attention projections, which influence how the model represents and uses the input. |
| Learning rate | 2e-4 | A practical learning rate for training the small adapter parameters while keeping the base weights frozen. |
| Epochs | 3 | Allows the adapters to learn the task while limiting unnecessary repeated training. |
| Per-device batch size | 4 | Kept the batch small to fit the Colab T4 GPU. |
| Gradient accumulation | 4 | Gives an effective batch size of 16 without placing all 16 examples in memory at once. |
| Maximum sequence length | 640 | Accommodates the prompt and target while keeping memory use manageable. |
| Weight decay | 0.01 | Provides additional regularization. |
| Warmup ratio | 0.03 | Gradually increases the learning rate at the start of training. |
| Scheduler | Cosine | Gradually reduces the learning rate during training. |
| Precision | FP16 | Reduces memory use and is supported by the T4 GPU. |
| Gradient checkpointing | Enabled | Saves activation memory by recomputing some activations during the backward pass. |

The effective batch size was:

`4 × 4 = 16`

The model had 496,195,456 parameters after adding the adapters. Of these, 2,162,688 were trainable (0.4359%); the original 494,032,768 base-model parameters remained frozen. This reduced the number of parameters that needed gradient and optimizer updates.

The notebook used the validation loss to select the best checkpoint. It saved the adapter and training history to Google Drive.

## 6. Training results

Training completed for approximately 3 epochs on the Tesla T4.

| Training measure | Result |
|---|---:|
| Training time | 4,466.13 seconds (about 74.4 minutes) |
| Training loss | 0.1191 |
| Best validation loss | 0.0764 |
| Trainable parameters | 2,162,688 |
| Trainable parameter share | 0.4359% |

The adapter was saved successfully. The notebook also saved the training history and training/validation loss plot.

## 7. Final test results

I loaded the saved adapter on top of the original base model and evaluated it on the same untouched test set of 3,080 examples.

| Metric | Base model | LoRA fine-tuned |
|---|---:|---:|
| Accuracy | 18.93% | **91.82%** |
| Macro F1 | 18.97% | **91.86%** |
| Weighted F1 | 18.97% | **91.86%** |
| Invalid prediction rate | 26.49% | **0.03%** |
| Correct predictions | 583 | **2,828** |

The LoRA model improved accuracy by **72.89 percentage points**. It produced 3,079 valid predictions out of 3,080; only one output was invalid.

Macro F1 gives equal importance to each intent class, while weighted F1 accounts for the number of examples in each class. Their similar values are consistent with the balanced test set, which has 40 examples per class.

The result shows that adapting a small fraction of the model parameters substantially improved its ability to follow the banking intent classification format and identify the correct intent.

## 8. Failure analysis

I inspected three incorrect predictions from the fine-tuned model. These examples are taken from the notebook's `lora_failure_examples.csv`.

| Customer query | Expected intent | Model output |
|---|---|---|
| I had a wrong fee charged at this ATM. | `cash_withdrawal_charge` | `wrong_charge` (invalid) |
| How do I locate my card? | `card_arrival` | `lost_or_stolen_card` |
| When will I get my card? | `card_arrival` | `card_delivery_estimate` |

**1. Invalid label generation**

For the ATM fee query, the model generated `wrong_charge`. This label is not in the Banking77 label set. The query is about a cash withdrawal fee, but the model produced a shortened, unsupported label. The output was therefore counted as invalid.

**2. Confusion between card arrival and a lost card**

For “How do I locate my card?”, the model predicted `lost_or_stolen_card`. The query refers to locating or tracking a card, while the expected class is `card_arrival`. These intents are related, but the model did not distinguish the delivery-status question from a lost-card report.

**3. Confusion between card arrival and delivery estimate**

For “When will I get my card?”, the model predicted `card_delivery_estimate` instead of `card_arrival`. Both labels concern card delivery. The model selected a closely related intent, but not the dataset's expected category.

These errors suggest that the remaining mistakes are mainly related to strict label selection and closely related intent categories. A possible next step would be to add targeted examples for confusing intent pairs and enforce label validation after generation. These changes were not part of this experiment.

## 9. Estimated GPU memory for a 7B model

This section is a theoretical estimate. The notebook experiment used LoRA on Qwen2.5-0.5B; it did **not** train a 7B model or run QLoRA.

For a fair comparison, I use the following assumptions:
- Model size: 7 billion parameters.
- Full fine-tuning and LoRA base weights: FP16 (2 bytes per parameter).
- Full fine-tuning uses FP16 weights and gradients, FP32 Adam optimizer states (first and second moments), and an FP32 master copy of the weights.
- For the LoRA estimate, assume a 32-layer, 4,096-hidden-size 7B model and rank 16 adapters on the query, key, value and output projections.
- For QLoRA, assume 4-bit base weights (0.5 bytes per parameter) and the same FP16 LoRA adapters.
- The estimates below cover model weights, gradients and optimizer state. They exclude activations, temporary buffers, CUDA context, allocator overhead and quantization metadata. Actual peak GPU memory will be higher.

### A. Full fine-tuning

All 7B parameters are trainable.

| Component | Calculation | Memory |
|---|---:|---:|
| FP16 weights | 7B × 2 bytes | 14 GB |
| FP16 gradients | 7B × 2 bytes | 14 GB |
| FP32 Adam first and second moments | 7B × 8 bytes | 56 GB |
| FP32 master weights | 7B × 4 bytes | 28 GB |
| **Estimated total** | **7B × 16 bytes** | **112 GB (about 104.3 GiB)** |

Full fine-tuning has the highest memory requirement because gradients and optimizer states are maintained for all model parameters.

### B. LoRA fine-tuning

The original 7B model is loaded in FP16 but kept frozen. Only the adapter parameters are trained.

For the assumed 32-layer architecture, the number of LoRA parameters for the four target projections is:

`32 layers × 4 projections × 2 × 4,096 × 16 = 16,777,216 parameters`

The factor of 2 represents the two low-rank matrices used by each LoRA update.

| Component | Calculation | Memory |
|---|---:|---:|
| Frozen FP16 base weights | 7B × 2 bytes | 14 GB |
| FP16 adapter weights, gradients, FP32 Adam states and master copy | 16.78M × 16 bytes | About 0.27 GB |
| **Estimated total** | **14 + 0.27 GB** | **About 14.27 GB (13.29 GiB)** |

LoRA saves memory because the base weights do not receive gradients or optimizer updates. Only the much smaller adapter parameters require training state.

### C. QLoRA fine-tuning

QLoRA stores the frozen base model in 4-bit form and trains LoRA adapters.

| Component | Calculation | Memory |
|---|---:|---:|
| 4-bit base weights | 7B × 0.5 bytes | 3.5 GB |
| FP16 adapter weights, gradients, FP32 Adam states and master copy | 16.78M × 16 bytes | About 0.27 GB |
| **Estimated total** | **3.5 + 0.27 GB** | **About 3.77 GB (3.51 GiB)** |

QLoRA has the lowest estimated memory requirement because the frozen base model is stored at 4 bits instead of FP16. Quantization metadata and runtime memory add some overhead in practice.

### Memory comparison

| Method | Estimated memory |
|---|---:|
| Full fine-tuning | 112 GB |
| LoRA | 14.27 GB |
| QLoRA | 3.77 GB |

These are approximate parameter-state estimates, not guaranteed peak GPU-memory requirements. Actual training also needs memory for activations and runtime operations. The adapter count used for LoRA and QLoRA is based on the stated 7B architecture assumptions; it is not measured from the Qwen2.5-0.5B model used in the experiment.

## 10. Conclusion

I fine-tuned `Qwen/Qwen2.5-0.5B-Instruct` on Banking77 using LoRA. Only 0.4359% of the parameters were trainable. On the held-out test set, accuracy increased from 18.93% to 91.82%, and macro F1 increased from 18.97% to 91.86%. Invalid outputs also fell from 26.49% to 0.03%.

The experiment demonstrates that LoRA can adapt a small open-weight model to a 77-class intent classification task without updating all base-model parameters. The remaining errors show that closely related intents and strict label formatting still need attention. The 7B memory figures are analytical estimates for comparison; only the 0.5B LoRA experiment was run.

## Files produced by the notebook

The notebook saved the base-model predictions and classification report, base confusion matrix, baseline metrics, LoRA adapter, training metrics and history, validation metrics, training/validation loss plot, LoRA test predictions and classification report, LoRA confusion matrix, comparison results and failure examples in Google Drive under `Banking77_LoRA/`.
