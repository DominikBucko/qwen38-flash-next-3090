# How it works: a 125 GB MoE checkpoint on one 24 GB GPU and 64 GB of RAM

Qwen3.8-Flash-Next is a sparse mixture-of-experts model: each of its 48 MoE layers has 512 routed experts, and
every token uses 10 of them. The checkpoint used here
([albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE))
stores the routed experts in INT4 (62.8 GB), a 51.2B-parameter per-layer embedding (PLE) table in FP8 (51.3 GB),
and 9.8 GB of attention, recurrent (GDN) and other dense weights. That is about five times the VRAM of an RTX 3090.

The runtime splits the model by how often each part is used:

| Where | What | Size |
|---|---|---:|
| GPU (24 GB) | Attention, GDN, shared experts, embeddings, LM head | 9.8 GB |
| | 32 "hot" experts per layer (the most frequently routed ones) | 3.9 GB |
| | MTP draft layer (it shares the embeddings with the target) | 1.6 GB |
| | INT8 KV cache: 141,504 tokens | 3.05 GB |
| | Prefill staging buffers, CUDA graphs, workspace | ~5 GB |
| RAM (56 GiB for the container) | Expert arena: 19,480 of the 23,040 cold experts, huge pages, pinned | 46.0 GiB |
| | Serving processes, page cache (PLE rows, NVMe experts) | ~10 GiB |
| NVMe | The FP8 PLE table, read in place from the checkpoint files | 51.3 GB |
| | The 3,560 least-used cold experts, read through the page cache | 8.4 GiB |

The GPU peaks at about 23.4 GB. The model weights are not changed, pruned or requantized: the GPU, the CPU and the
prefill path all compute the same INT4 group-128 experts.

## Decode: the CPU computes the cold experts

A decode step verifies 4 tokens (1 + 3 MTP draft tokens). For each MoE layer:

1. The router picks 10 experts per token. Those among the layer's 32 hot experts run on the GPU as usual.
2. The GPU writes the other token-expert pairs into a small request slot in pinned host memory.
3. A pool of CPU threads (one per physical core, pinned) computes those cold experts straight from the RAM arena:
   an AVX2/FMA kernel that reads the INT4 weights and F16 scales, with FP32 activations and accumulation.
4. Meanwhile the GPU computes about 22% of the cold experts itself: a Triton kernel reads their INT4 weights from
   the same pinned arena over PCIe (about 25 GB/s) instead of waiting idle.
5. The GPU waits for the CPU's partial sum and adds both parts to the layer output.

The request/response slots are polled by kernels, so the whole step still runs as one CUDA graph. The CPU part
reads about 2 GB of expert weights per step (roughly 750–950 distinct cold experts), so **decode speed depends
mostly on CPU memory bandwidth**. On the benchmark host the CPU part takes 33–36 ms per step, the GPU part about
16 ms, and a step produces about 3 tokens.

Cold experts that are not in RAM (the NVMe "tail") are read through a file mapping. The arena is dynamic: after
each step up to 8 tail experts that decode just used move into their layer's least recently used arena slot, and
the routing statistics of the prompt suggest more (the experts a long prompt used most). The 200 best-ranked
experts per layer never move, so the GPU share can read them at fixed addresses.

## Prefill: stream the cold experts to the GPU

Prompt chunks of 8,192 tokens use every expert of every layer, so the CPU would be far too slow. Instead, for each
layer:

1. The hot experts run from VRAM.
2. The cold experts are copied from the pinned arena to the GPU by DMA, 80 at a time (one contiguous range each),
   two groups ahead of the compute, at about 27 GB/s.
3. On the GPU each group is converted into the Humming kernel layout, byte-identical to what the model loader
   produces for the hot experts, and multiplied with the original indexed GEMM kernels.
4. All groups add into one FP32 accumulator, rounded to BF16 once at the end.

Tail experts are read from NVMe a few layers ahead by a small C++ thread pool (one call per layer, outside
Python's GIL, so the thread that launches the GPU kernels is never held up), into pinned bounce buffers, and
copied on a separate stream. A full pass over the cold experts costs about 2.2 s of PCIe time per chunk, which
overlaps with the attention and GDN work. Small chunks (up to 384 tokens, such as the end of a prompt) run on
the CPU path instead of streaming all experts.

## KV cache and context

The attention KV cache is INT8 with one scale per token and KV head, dequantized inside the sparse attention
(QSA) kernel. That fits 141,504 tokens in 3.05 GB, enough for one 135,168-token request plus the recurrent-state
blocks that async prefill holds. The recurrent (GDN) state stays in FP32.

## PLE table from NVMe

The 51.2B-parameter FP8 PLE table is never loaded. Its checkpoint shards are memory-mapped read-only, and each
step gathers only the rows its tokens need (with read-ahead hints). The page cache keeps the recently used rows.

## What stays the same as the 2× RTX 3090 runtime

This runtime is an overlay on the [2× RTX 3090 v0.4.0 release](https://github.com/DominikBucko/qwen38-flash-next-2x3090):
same pinned vLLM build, same checkpoint, same Humming GPU kernels, MTP3 speculative decoding, prefix caching,
tool calling and reasoning parsers. What differs:

| | 2× RTX 3090, agent 128K profile | 1× RTX 3090, this repository |
|---|---|---|
| GPUs | 2 (tensor + expert parallel) | 1 |
| System RAM | 128 GB | 64 GB |
| Cold experts computed on | GPU (copied from RAM) | CPU (decode), GPU (streamed, prefill) |
| KV cache | BF16 | INT8 per token and head |
| PLE table | In RAM | On NVMe |
| Context | 135,168 tokens | 135,168 tokens |
| Prefill, 131K prompt | 3,029 tok/s | 2,106 tok/s |
| Decode | ~110 tok/s | 47–50 tok/s |

## Numerical checks

- CPU kernel vs a NumPy FP32 dequantize-and-matmul reference: max relative error 1.5e-6.
- GPU share kernel vs an FP32 reference: 3.1e-4 relative error (FP16 dot products, FP32 accumulation).
- Prefill conversion vs the loader's conversion of the same experts: byte-identical.
- INT8 KV: greedy outputs drift from BF16 KV after a median of 13–29 tokens on 10 prompts, the same range as two
  runs of the same BF16 configuration (23 tokens), because the runtime is not bit-deterministic. A
  teacher-forced likelihood comparison has not been run yet.
