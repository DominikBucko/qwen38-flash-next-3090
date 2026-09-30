# Tuning and troubleshooting

The defaults live in [`configs/3090-64gb-128k.env`](../configs/3090-64gb-128k.env). Put overrides in `.env`;
`make serve` passes them to the container. Change one thing at a time and compare with `make bench`.

## Memory

| Setting | Default | Meaning |
|---|---|---|
| `MEMORY_LIMIT` | `auto` | Memory limit of the whole container. `auto` = installed RAM (MemTotal rounded up to a multiple of 8 GiB) minus `HOST_RESERVE_GIB`: 56 GiB on a 64 GB machine. |
| `HOST_RESERVE_GIB` | `8` | RAM left for the OS and desktop when `MEMORY_LIMIT=auto`. |
| `QWEN38_EXPERT_ARENA_SLOTS` | `auto` | Cold experts kept in RAM. `auto` = (container limit − 10 GiB) / 2.42 MiB, at most all 23,040. The rest are read from NVMe. |
| `QWEN38_SERVE_OVERHEAD_GIB` | `10` | RAM that `auto` leaves inside the container for the processes and the page cache. |

With 64 GB of RAM, 19,480 of the 23,040 cold experts fit in RAM (46 GiB) and 3,560 stay on NVMe. With 96 GB or
more, every cold expert fits and decode never waits for NVMe. With 48 GB, about 12,700 fit; it should run, but
decode reads more experts from NVMe (not measured).

The arena asks for transparent huge pages (`madvise` mode is enough). On a freshly booted machine all of it gets
huge pages; after days of uptime memory can be fragmented. The startup log line `Expert store: ... huge pages
X GiB` shows how much you got. `sync; echo 1 | sudo tee /proc/sys/vm/compact_memory` before starting helps.

## GPU memory

The default profile peaks at about 23.4 GB of the 3090's 24 GB. If the same card drives your display, check
`nvidia-smi` first: the desktop typically holds 0.3–1.5 GB. If the server runs out of GPU memory, free VRAM with
one of these, in this order:

1. `QWEN38_HOT_ONLY=28` (from 32): each hot expert per layer costs ~116 MiB of VRAM (48 layers × 2.42 MiB),
   so 4 fewer free ~465 MiB. Fewer hot experts means more work for the CPU per decode step.
2. A shorter context: `MAX_MODEL_LEN=65536` with `KV_CACHE_MEMORY_BYTES=2300000000` frees ~0.75 GB. Measured
   capacities with the INT8 cache: 3.05 GB holds 141,504 tokens, 2.0 GB holds 49,152. Keep the capacity that
   the startup log reports (`GPU KV cache size`) at least 1.05× `MAX_MODEL_LEN`: async prefill holds extra
   recurrent-state blocks.
3. `MAX_NUM_BATCHED_TOKENS=4096` (from 8192): smaller prefill activations, slower prefill.

These alternatives are not benchmarked.

## CPU

| Setting | Default | Meaning |
|---|---|---|
| `QWEN38_CPU_EXPERTS_CPUS` | `auto` | CPUs of the cold-expert pool (one thread each). `auto`: one thread per physical core except the first quarter of the cores of every L3 domain (CCD), at most 24. |
| `QWEN38_CPU_EXPERTS_MAX_THREADS` | `24` | Cap for `auto`. |
| `QWEN38_MAIN_CPUS` | `auto` | CPUs of the serving processes: everything except the SMT siblings of the pool's cores. |
| `CPUSET` | unset | Restrict the whole container to these CPUs (`docker --cpuset-cpus`). |
| `QWEN38_PLE_THREADS` | `4` | CPU threads for the embedding-table lookups. More threads only spin and take cores from the GPU worker. |

The container prints its plan at startup, for example on a 16-core Ryzen 9 7950X:

```text
[qwen38-3090] CPU expert pool: 12 threads on CPUs 2-7,10-15
[qwen38-3090] Serving processes: CPUs 0-17,24-25
```

Decode is limited by how fast the pool can read ~2 GB of expert weights per step. More memory bandwidth helps
most: dual-channel DDR5 at 6000 MT/s or faster, both channels populated (two or four DIMMs), EXPO/XMP enabled.
On AMD, each CCD has its own link to memory, so the pool should span every CCD (`auto` does). On Intel hybrid
CPUs, `auto` keeps the first P-cores for the serving processes and gives the pool the remaining P- and E-cores;
work is distributed dynamically, so slower cores simply take fewer experts.

Prefill needs CPU time too (embedding lookups, NVMe reads, launching GPU kernels): the pool sleeps during long
prefill steps and the serving processes may use its cores. On the benchmark CPU restricted to 16 cores, prefill
matched the full 32 cores once the embedding lookups were limited to 4 threads (before, they took ~13 cores
and halved long-prompt prefill).

## Decode

| Setting | Default | Meaning |
|---|---|---|
| `QWEN38_GPU_SHARE_FRAC` | `0.22` | Share of each layer's cold experts the GPU computes over PCIe while the CPU works. On the benchmark host 0.27 made the GPU the bottleneck; a CPU with less memory bandwidth may gain from a higher value (untested). `0` turns it off. |
| `QWEN38_EXPERT_PROMOTE` | `8` | NVMe experts moved into RAM per decode step. `0` = static arena. |
| `MTP_DEPTH` | `3` | Draft tokens per step (0–3). The CPU path handles at most 4 tokens per step. |
| `QWEN38_CPU_EXPERTS_STATS` | `0` | Log decode step timings every N steps: CPU ms, GPU gap, NVMe experts, promotions. |

With `QWEN38_CPU_EXPERTS_STATS=64` the server logs a line like this every 64 decode steps:

```text
cpu_moe decode steps=64: per step cpu 32.80 ms, between layers 15.83 ms, step tail 4.63 ms (48.0 requests/step),
cold experts 772/step of which tail 1.2; promoted 3935, skipped 2126 (total)
```

`cpu` is the CPU pool's time per step and `between layers` the GPU's. If `cpu` is well above the GPU time, raise
`QWEN38_GPU_SHARE_FRAC` a little; if the GPU time grows past the CPU time, lower it. `of which tail` counts
experts read from NVMe: large values mean the RAM arena is too small for the workload.

## Prefill

| Setting | Default | Meaning |
|---|---|---|
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Prefill chunk size. |
| `QWEN38_STREAM_GROUP` | `80` | Cold experts per DMA group. Larger groups need more VRAM. |
| `QWEN38_CPU_PREFILL_MAX` | `384` | Chunks up to this many tokens run on the CPU instead of streaming every expert. |
| `QWEN38_STREAM_PROFILE` | `0` | `1` logs a per-step breakdown (DMA stalls, conversion, GEMMs). |
| `QWEN38_TAIL_BUFFERS` | `4` | Pinned bounce buffers (one layer of NVMe experts each, ~188 MB) that prefill reads ahead into. |

Each 8,192-token chunk also reads the NVMe-tier experts (about 9 GB with 64 GB of RAM). On the benchmark host's
PCIe 3.0 SSD that is ~2.5 s per chunk, mostly hidden behind the GPU work; a PCIe 4.0 SSD, or more RAM (fewer
NVMe experts), shortens it.

The first request after a start is slower (kernel compilation and first-touch page faults), and so are the first
~64 decode steps after a new long prompt, while the arena adapts to it.

## Other

| Setting | Default | Meaning |
|---|---|---|
| `MAX_MODEL_LEN` | `135168` | Tokens per request (prompt + output). |
| `MAX_NUM_SEQS` | `1` | Must stay 1: requests are served one at a time and queue in the server. |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | `3200` | Prefix-cache checkpoint interval; a multiple of 3,200 with the INT8 KV cache. |
| `KV_CACHE_DTYPE` | `int8_per_token_head` | `auto` gives BF16 KV (half the tokens per GB). |
| `VLLM_API_KEY` | unset | Require this bearer token on the API. |
| `JIT_CACHE_DIR` | `./jit-cache` | Humming and Triton kernel caches, kept between starts. |

## Common problems

- **`docker: could not select device driver "" with capabilities: [[gpu]]`**: install the NVIDIA Container
  Toolkit and restart Docker.
- **The container exits during `Expert store`** with an out-of-memory kill: the memory limit is larger than what
  is actually free. Close other programs, or set `MEMORY_LIMIT` lower (the arena shrinks with it).
- **CUDA out of memory at startup or during the first long prompt**: see [GPU memory](#gpu-memory).
- **Illegal instruction**: the CPU lacks AVX2/FMA/F16C (pre-2013 Intel, pre-2015 AMD).
- **Very slow decode (under 20 tok/s)**: check that the checkpoint is on an NVMe SSD, that RAM runs in dual
  channel at its rated speed, and that nothing else is using the memory bandwidth. Set
  `QWEN38_CPU_EXPERTS_STATS=64` and look at `tail experts/step`: large values mean the arena is too small.
