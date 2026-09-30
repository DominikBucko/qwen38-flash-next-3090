# FAQ

### Can I run Qwen3.8-Flash-Next on a single RTX 3090?

Yes. This repository serves it on one RTX 3090 (24 GB) with 64 GB of system RAM, with a 128K context
(135,168 tokens per request), through an OpenAI-compatible API. The GPU holds the dense weights, 32 hot experts
per layer and an INT8 KV cache; the CPU computes the other experts during decode. See
[how it works](how-it-works.md).

### How much RAM do I need? Does 32 GB work?

64 GB is the tested size. The runtime keeps 46 GiB of cold experts in RAM and the rest on NVMe. With 96 GB or
more every expert fits in RAM. 48 GB should start, with more NVMe reads during decode (untested). 32 GB is not
enough.

### Does it work on an RTX 4090, 3090 Ti, A5000 or another 24 GB card?

It should, but only the RTX 3090 is tested. The runtime needs about 23.4 GB of VRAM. Please report results with
the [hardware report form](https://github.com/DominikBucko/qwen38-flash-next-3090/issues/new?template=hardware-report.yml).
16 GB cards (RTX 4080, 4070 Ti Super, 4060 Ti 16 GB) do not fit the default profile.

### How fast is it?

On the benchmark machine (limited to 64 GB of RAM), a 131,072-token prompt prefills at 1,450–2,100 tokens per
second (62–90 s to the first token, depending on a [known issue](../benchmarks/2026-09-30/README.md#known-issue-occasional-slow-prefill-steps)),
and decode runs at 43–51 tokens per second. With 16 or 8 cores of the same CPU, decode drops to about 45 and 35
tok/s. See the [README](../README.md#results) and the
[benchmark report](../benchmarks/2026-09-30/README.md). Decode depends mostly on your RAM bandwidth.

### Which CPU do I need?

Any x86-64 CPU with AVX2 (Ryzen, Intel Core since 2013). The CPU computes the cold experts during decode, so
memory bandwidth matters more than core count: dual-channel DDR5 is the sweet spot. The runtime picks the cores
itself; see [CPU tuning](tuning.md#cpu).

### Is the output quality the same as on two GPUs?

The weights, the INT4 expert math and the sparse attention are the same as in the
[2× RTX 3090 runtime](https://github.com/DominikBucko/qwen38-flash-next-2x3090). The difference is the KV cache:
INT8 per token and head instead of BF16. In a greedy-decoding comparison INT8 and BF16 differ no more than two
BF16 runs differ from each other; a likelihood-based comparison has not been run yet.

### How is this different from llama.cpp `--n-cpu-moe` or KTransformers?

Same basic idea: keep the busy parts of a mixture-of-experts model on the GPU and let the CPU handle experts
that do not fit. This runtime is built on vLLM for this specific model and checkpoint: it keeps the most
frequently routed experts of every layer on the GPU, computes the rest on the CPU during decode (with the GPU
helping over PCIe), and streams all experts through the GPU during prefill, so long prompts stay fast. It uses
the model's MTP draft layer for speculative decoding and its sparse attention kernels. There is no GGUF version;
it runs the Intel AutoRound INT4 checkpoint directly.

### Can it serve several users or parallel requests?

It serves one request at a time; further requests wait in the queue. The CPU decode path handles one sequence
of up to 4 tokens per step.

### Does it support tool calling, reasoning and a long context?

Yes: OpenAI-compatible `/v1/chat/completions` and `/v1/completions`, tool calls (`qwen3_coder` parser),
reasoning output (`qwen3` parser, `reasoning_content`), prefix caching, and up to 135,168 tokens per request.
Point any OpenAI-compatible client at `http://127.0.0.1:8000/v1`.

### Why is the first request slow?

The first start compiles GPU kernels (they are cached in `./jit-cache` for later starts), and the first
requests touch memory for the first time. After a new long prompt, the first ~64 decode steps are also slower
while the RAM arena adapts to the experts that prompt uses.

### How long does startup take, and how big is the download?

The checkpoint is about 129 GB (target plus MTP draft). Startup takes about 3–4 minutes with warm kernel caches;
the first start takes longer.

### Does it run on Windows or WSL2?

It is only tested on Linux. WSL2 with Docker and GPU support may work, but its memory limits and huge-page
behaviour differ.

### Can I use a SATA SSD or a hard disk?

The checkpoint is read during serving (PLE rows and the least-used experts), so use an NVMe SSD. A SATA SSD will
work but slower; a hard disk is not usable.
