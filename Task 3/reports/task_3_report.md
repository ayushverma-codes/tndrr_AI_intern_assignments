# Task 3: Running a Model Locally & Inference Optimisation

## 1. Objective

The goal of this task was to run a small open-source language model, compare two quantisation levels, and understand how quantisation affects inference speed, memory usage, and output quality.

For this experiment, I used **Qwen2.5-1.5B-Instruct** with two quantisation levels:

- **Q4_K_M:** 4-bit quantisation
- **Q8_0:** 8-bit quantisation

I compared both versions using the same 10 prompts and recorded their generation speed, GPU memory usage, and responses.

I also studied the KV cache and estimated how I would serve 20 concurrent users on a single 24 GB GPU.

## 2. Why I Used Google Colab

My laptop does not have a dedicated GPU and is already slow, so running an LLM locally could make it less responsive.

Since the assignment allows using a free GPU, I chose **Google Colab**.

Colab provided an NVIDIA Tesla T4 GPU with 15 GiB of VRAM. This allowed me to run both models using GPU acceleration without putting extra load on my laptop.

I used Ollama to download and run the models. I also verified that the model was running with **100% GPU offloading** using `ollama ps`.

## 3. Experimental Setup

| Component | Details |
|---|---|
| Platform | Google Colab (Free) |
| GPU | NVIDIA Tesla T4 |
| GPU memory | 15 GiB |
| Inference framework | Ollama 0.40.16 |
| Inference backend | llama.cpp (through Ollama) |
| Model | Qwen2.5-1.5B-Instruct |
| Quantisation levels | Q4_K_M and Q8_0 |
| Number of prompts | 10 |
| Repetitions per prompt | 2 |
| Total benchmark runs | 40 |
| Context length | 4096 tokens |
| Maximum output tokens | 256 |
| Temperature | 0 |
| Random seed | 42 |

### How I Conducted the Experiment

1. Installed Ollama on Google Colab and started its server.
2. Downloaded both quantised versions of Qwen2.5-1.5B-Instruct.
3. Verified that inference was running on the T4 GPU.
4. Prepared 10 fixed prompts covering different tasks.
5. Ran each prompt twice on each model.
6. Recorded generation speed and GPU memory usage.
7. Compared the responses from both versions.
8. Ran an additional validation test on two prompts where the original answers were cut off or incomplete.

I used the same generation settings for both models to keep the comparison fair.

Each model was warmed up before benchmarking, and the other model was unloaded to avoid interference.

## 4. Performance Results

### 4.1 Tokens per Second and GPU Memory

| Metric | 4-bit (Q4_K_M) | 8-bit (Q8_0) |
|---|---:|---:|
| Average generation speed | 137.12 tokens/sec | 99.89 tokens/sec |
| Median generation speed | 137.61 tokens/sec | 99.46 tokens/sec |
| Maximum observed GPU memory | 1,243 MiB | 1,873 MiB |
| Completed benchmark runs | 20 | 20 |

The values above come from the actual benchmark output.

### 4.2 Speed Comparison

The 4-bit model achieved an average speed of **137.12 tokens/sec**, compared with **99.89 tokens/sec** for the 8-bit model.

Speed improvement:

\[
\frac{137.12-99.89}{99.89}\times100
\approx37.3\%
\]

So, the 4-bit model generated tokens approximately **37.3% faster** than the 8-bit model.

### 4.3 Memory Comparison

The 4-bit model used a maximum observed GPU memory of 1,243 MiB, while the 8-bit model used 1,873 MiB.

Memory reduction:

\[
\frac{1873-1243}{1873}\times100
\approx33.6\%
\]

The 4-bit version used approximately **33.6% less GPU memory**, saving 630 MiB.

GPU memory was measured by repeatedly sampling `nvidia-smi`. These are observed memory values, so very short memory spikes might not have been captured.

The speed measurements represent token generation speed, not the complete time required to load a model and produce a response.

### Observation

The 4-bit model was both faster and more memory-efficient in this experiment.

This makes 4-bit quantisation attractive when GPU memory is limited or inference speed is important.

However, better speed does not automatically mean better output quality. That is why I also compared the responses.

## 5. Output Quality Comparison

I tested both models on the same 10 prompts.

The prompts covered mathematics, logical reasoning, Python programming, debugging, summarisation, conceptual explanations, instruction following, JSON formatting, arithmetic, and safety.

I compared the first response from each model based on correctness, completeness, and whether it followed the instructions.

### 5.1 Results on the 10 Prompts

| Prompt | Task | 4-bit Result | 8-bit Result |
|---|---|---|---|
| P01 | Mathematics | Correct (439) | Correct (439) |
| P02 | Logical reasoning | Incorrect conclusion | Same incorrect conclusion |
| P03 | Python palindrome | Working function, but incorrect example involving punctuation | Working function that also handles punctuation |
| P04 | Debugging | Correctly handles an empty list | Correctly handles an empty list |
| P05 | Summarisation | Correct and concise | Correct and concise |
| P06 | ML explanation | Correct explanation | Correct explanation |
| P07 | Caching benefits | Followed format, but made a questionable claim | Same issue |
| P08 | JSON formatting | Correct fields, but added Markdown fences | Same issue |
| P09 | Discount calculation | Initially cut off; correct in validation | Correct (₹660) |
| P10 | Password safety | Appropriate response | Appropriate response |

### 5.2 Important Observations

**1. Both models performed similarly on simple tasks.**

For mathematics, summarisation, ML concepts, and debugging, both models produced correct answers.

For example, both correctly calculated:

`17 × 23 + 48 = 439`

**2. Both models made the same reasoning mistake.**

In the logic question, the models were told that all roses are flowers and some flowers fade quickly.

Both incorrectly concluded that some roses must fade quickly.

This conclusion does not necessarily follow because the flowers that fade quickly may not include roses.

This shows that higher precision does not automatically fix reasoning mistakes.

**3. The 8-bit model gave a better Python example.**

Both models wrote a palindrome-checking function that ignored spaces and capitalisation, as requested.

However, the 4-bit model claimed its code would correctly handle a sentence containing punctuation, even though its code did not remove punctuation.

The 8-bit model also removed punctuation and handled that example correctly.

**4. Both models had instruction-following issues.**

In the JSON task, both returned the correct data but wrapped it in Markdown code fences, despite being asked to return only JSON.

Similarly, both provided three numbered caching benefits, but incorrectly suggested that caching generally reduces storage requirements.

**5. Output length affected one result.**

In the original arithmetic test, the 4-bit model reached the 256-token output limit before giving the final answer.

To check whether this was a model-quality issue, I repeated the arithmetic test for both models with a 512-token limit.

Both correctly calculated:

- Original price: ₹750
- Discounted price: ₹600
- Tax: ₹60
- Final price: **₹660**

I also repeated the Python prompt with the higher limit. Both models completed their responses, but the 4-bit model still contained the incorrect punctuation-related example.

These were additional validation tests, not part of the original 40-run speed benchmark.

### 5.3 Quality Trade-off

The overall response quality was similar for most prompts.

The 8-bit model performed better on the Python example, while both models shared several reasoning and formatting mistakes.

The 4-bit model produced the correct arithmetic answer when given enough output tokens.

Based on these 10 prompts, I did not observe a large, consistent quality difference between the two quantisation levels.

However, this is a small evaluation, so it is not enough to prove that 4-bit and 8-bit models have equal accuracy across all tasks.

## 6. What Is the KV Cache?

When a language model generates text, it predicts one token at a time.

To generate the next token, the transformer uses attention to look at information from earlier tokens.

Without caching, it would need to calculate the keys and values of earlier tokens again and again.

The **KV cache stores these previously calculated keys and values**, so the model can reuse them while generating new tokens.

For example, if the model has already processed 100 tokens, it can reuse their stored keys and values when generating token 101 instead of calculating them all again.

This helps speed up text generation.

### The Trade-off

The KV cache makes generation faster, but it also consumes GPU memory.

As the context becomes longer or more users make requests, the KV cache grows.

So, when serving multiple users, we need to consider both model memory and KV-cache memory.

## 7. Why Quantisation Saves Memory

Model weights are normally stored using numerical formats such as FP16, which uses 16 bits per value.

Quantisation represents weights using fewer bits.

For example:

| Precision | Approximate storage for 1 billion weights |
|---|---:|
| FP16 (16-bit) | 2 GB |
| INT8 (8-bit) | 1 GB |
| INT4 (4-bit) | 0.5 GB |

These are ideal raw weight-storage estimates. Actual quantised models also contain scaling information and other metadata.

Using fewer bits reduces the memory required to store the model weights.

It can also reduce the amount of data transferred from GPU memory during inference, which may improve speed.

### Trade-off

Lower-bit quantisation saves memory, but it introduces some approximation because weights are stored with less precision.

This can affect the model's outputs.

Also, lower precision does not always mean faster inference. Speed depends on GPU hardware, quantisation format, software support, and workload.

In my experiment:

- 4-bit was faster.
- 4-bit used less GPU memory.
- Both models produced similar answers on most prompts.
- 8-bit handled one Python example better.

This shows why **speed, memory usage, and output quality should be evaluated together**.

## 8. How I Would Serve 20 Users on One 24 GB GPU

This section is a back-of-envelope estimate, not an actual deployment.

I did not have access to a 24 GB GPU. The experiment was performed on a Tesla T4, so the following calculations describe how I would approach this problem.

### 8.1 Assumptions

| Parameter | Assumption |
|---|---|
| GPU | One GPU with 24 GB VRAM |
| Model | Qwen2.5-7B-Instruct |
| Quantisation | 4-bit weights |
| Serving framework | vLLM |
| Concurrent active users | 20 |
| Maximum total context per request | 4,096 tokens |
| Maximum generated output | 256 tokens |
| KV-cache precision | FP16 |
| Deployment | One shared model instance |

I would choose a 4-bit model to reduce model memory usage and leave more GPU memory for active requests.

I would use vLLM because it supports continuous batching and efficient KV-cache management.

### 8.2 KV-Cache Memory Estimate

For Qwen2.5-7B-Instruct, I assume:

- 28 transformer layers
- 4 KV attention heads
- 128 dimensions per head
- 2 bytes per value (FP16)

KV-cache memory per token:

\[
2\times28\times4\times128\times2
=57,344\text{ bytes}
\]

This is approximately 56 KiB per token.

For 20 users, each using the full 4,096-token context:

\[
20\times4096\times56\text{ KiB}
\]

\[
\boxed{4.375\text{ GiB}}
\]

So, approximately **4.375 GiB of GPU memory** would be needed for the KV cache in this worst-case context-length estimate, excluding allocation overhead.

### 8.3 GPU Memory Budget

A GPU advertised with 24 GB of memory has approximately 22.35 GiB of capacity.

My estimated allocation would be:

| Component | Estimated Memory |
|---|---:|
| 4-bit model weights and related storage | 5.5 GiB |
| KV cache for 20 requests | 4.375 GiB |
| Runtime, activations and temporary buffers | 4.0 GiB |
| Remaining memory | Approximately 8.5 GiB |

These are assumptions for planning, not measured memory figures.

The available memory suggests that 20 active sequences may be possible, but actual memory usage would need to be checked after loading the model and running concurrent requests.

### 8.4 Serving Approach

I would use the following approach:

1. **Load one model:** All users share the same model instead of loading a separate copy for each user.
2. **Use continuous batching:** Combine work from multiple active requests to improve GPU utilisation.
3. **Manage the KV cache efficiently:** Use vLLM's paged KV-cache management to reduce memory waste.
4. **Limit context and output length:** Prevent individual requests from consuming too much GPU memory.
5. **Use a request queue:** If the GPU cannot handle more active requests without unacceptable latency, queue additional requests.

### 8.5 Expected Speed for 20 Users

Even if 20 users fit in GPU memory, they may not all receive responses as quickly as a single user.

For a simple estimate, assume the GPU achieves an aggregate decoding throughput of 200 tokens per second while serving all 20 users.

Average throughput per user:

\[
200/20=10\text{ tokens/sec}
\]

For a response of 256 tokens:

\[
256/10=25.6\text{ seconds}
\]

This is only an example based on assumed throughput. It excludes prompt processing, queuing, and networking delays.

I would verify the actual performance by testing 1, 5, 10, and 20 concurrent requests and measuring GPU memory, total throughput, and response latency.

**My conclusion is that fitting 20 requests in memory is only the first step. Serving all 20 users with acceptable response times also requires proper batching and load testing.**

## 9. Limitations

There are a few limitations to this experiment:

1. I tested only one model family with two quantisation levels.
2. I used 10 prompts, which is a small sample for evaluating overall model quality.
3. The original 256-token output limit caused truncation in some responses.
4. GPU memory was measured through sampling rather than exact peak allocation tracking.
5. Performance was measured for sequential requests, not 20 simultaneous users.
6. The 24 GB GPU serving plan is an estimate and was not tested on actual hardware.

These limitations are important when interpreting the results.

## 10. Final Conclusion

In this experiment, I ran Qwen2.5-1.5B-Instruct on a free Google Colab Tesla T4 GPU using Ollama and compared its 4-bit and 8-bit versions.

The main results were:

| Metric | 4-bit | 8-bit |
|---|---:|---:|
| Average speed | 137.12 tok/s | 99.89 tok/s |
| Observed GPU memory | 1,243 MiB | 1,873 MiB |

The 4-bit model was approximately **37.3% faster** and used **33.6% less GPU memory**.

For output quality, both models performed similarly on most tasks, although the 8-bit model gave a better Python example.

The additional validation also showed that output length can affect quality comparisons, as the 4-bit model correctly solved the arithmetic question when given enough tokens.

**For this tested workload, I would prefer the 4-bit model because it provides better speed and memory efficiency with no large, consistent loss in quality observed across the 10 prompts.**

The main thing I learned is that choosing a model is not only about accuracy. We also need to consider memory usage, inference speed, response quality, and the number of users the system needs to support.

---

## References and Reproducibility

**Experiment files:**
- `Task_3.ipynb` — Colab notebook containing installation, model execution, benchmarking, and validation
- `benchmark_detailed.csv` — Detailed performance results from 40 runs
- `benchmark_summary.csv` — Summary of throughput and GPU memory
- `quality_comparison.csv` — Responses to the 10 prompts from both models
- `quality_validation.csv` — Additional validation for the Python and arithmetic prompts

**Tools and documentation:**
- [Ollama](https://ollama.com/)
- [Qwen2.5 Models](https://huggingface.co/Qwen)
- [Qwen2.5-7B-Instruct Model Configuration](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Google Colab](https://colab.research.google.com/)