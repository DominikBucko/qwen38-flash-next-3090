# One RTX 3090 + 64 GB RAM: benchmarks, September 30, 2026

**2,106 tok/s prefill on a 131K-token prompt and 47–50 tok/s decode**, with the default
[`configs/3090-64gb-128k.env`](../../configs/3090-64gb-128k.env) profile on one RTX 3090 and a machine limited to
64 GB of RAM. The images were built with [`docker/Dockerfile`](../../docker/Dockerfile) from this tree and started
with [`scripts/docker_serve.sh`](../../scripts/docker_serve.sh).

## Setup

| | |
|---|---|
| GPU | 1× RTX 3090 24 GB, PCIe 4.0 ×16 |
| CPU | AMD Threadripper PRO 5975WX (32 cores, 4 CCDs, Zen 3) |
| RAM | 8×16 GB DDR4-3200; **64 GiB usable**: the RAM above 64 GiB was locked by [`tools/ram-balloon.sh`](../../tools/ram-balloon.sh), started after the server's expert arena was populated; server container limit 56 GiB |
| NVMe | Samsung PM981a (PCIe 3.0 ×4), also the system disk |
| Software | Linux 7.0, NVIDIA driver 595.84, Docker |
| Runtime | This tree: 19,480 cold experts in RAM (46.0 GiB, all on huge pages), 3,560 on NVMe; CPU pool chosen automatically |

Protocol ([`scripts/bench.py`](../../scripts/bench.py) with its defaults): after a fresh server start with warm
kernel caches, four requests run one at a time in this order: 4,096 + 256, 32,768 + 512, 131,072 + 512 and
8,192 + 1,024 tokens (input + output). Prompts are code-review requests built from this repository's runtime
sources, greedy (temperature 0), a unique `cache_salt` each (no prefix-cache hits), outputs forced to the full
length. Prefill = prompt tokens / time to first token; decode = (output tokens − 1) / time after the first
token. MTP speculative decoding with 3 draft tokens: 1.9–2.3 of 3 draft tokens were accepted per step in all
runs. No request was preempted.

## Results, release image

32 cores (the whole benchmark CPU; automatic CPU pool: 24 threads):

| Request (input + output) | First token | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| 131,099 + 512 | 62.2 s | **2,106** | **47.4** |
| 32,799 + 512 | 15.3 s | 2,141 | 50.3 |
| 8,218 + 1,024 | 5.2 s | 1,570 | 49.1 |
| 4,127 + 256, first request after the start | 3.3 s | 1,258 | 47.8 |

Peaks: GPU memory 23,369 MiB of 24,576, GPU 67–68 °C, container memory at its 56.0 GiB limit (52–53 GiB
anonymous, of which the expert arena is 46.0 GiB on huge pages; the rest is page cache), CPU Tctl 86–88 °C.

Smoke test ([`scripts/smoke_test.py`](../../scripts/smoke_test.py)) after every run: a reasoning answer (17 × 23
= 391), a `get_weather` tool call with the right argument, and a Python function with thinking off, all passed.

### Desktop CPU proxies

Decode runs the cold experts on the CPU, so it depends on how fast the CPU reads memory. To see how it scales,
the whole server container was restricted with `CPUSET` to part of the benchmark CPU; the runtime's automatic CPU
plan then picked the pool shown. These are proxies: the same Zen 3 cores and DDR4 memory, where each CCD's link
to memory limits bandwidth (20–30 GB/s per CCD in an isolated kernel test). Desktop Zen 4/5 CCDs and Intel
desktop CPUs with DDR5 have more bandwidth per core, so real desktops may do better.

| Container CPUs | Like | CPU pool | 4K + 256 | 32K + 512 | 131K + 512 | 8K + 1,024 |
|---|---|---|---|---|---|---|
| 32 cores, 4 CCDs | the benchmark machine | 24 threads | 1,258 / 47.8 | 2,141 / 50.3 | 2,106 / 47.4 | 1,570 / 49.1 |
| 16 cores, 2 CCDs (`0-15,32-47`), run 1 | Ryzen 9 7950X / 9950X | 12 threads | 1,200 / 45.2 | 2,192 / 44.9 | 1,452 / 39.3 | 2,026 / 46.2 |
| 16 cores, 2 CCDs, run 2 | | 12 threads | 1,268 / 42.1 | 2,102 / 44.9 | 2,104 / 40.5 | 1,419 / 46.8 |
| 8 cores, 1 CCD (`0-7,32-39`), earlier image | Ryzen 7 7800X3D / 9700X | 6 threads | 1,108 / 35.5 | 2,133 / 35.5 | 2,149 / 33.6 | 1,951 / 34.6 |

(prefill / decode tok/s). With fewer cores decode drops (about 45 tok/s on 16 cores, 35 on 8), and prefill
stays at ~2,100 tok/s when it takes the fast path.

## Known issue: occasional slow prefill steps

A prefill chunk of 8,192 tokens normally takes ~3.7 s. In some requests every chunk takes ~5.5 s instead: the
NVMe reads of the experts that are not in RAM and the GPU work stop overlapping. It hits the single-chunk 8K
request in about half of all runs (5.2–6.0 s instead of 4.0–4.3 s to the first token). On the 16-core proxy it
hit the 131K request in 5 of 6 runs without the prefill event markers (`QWEN38_STREAM_MARKERS`) and in 1 of 6
with them, so the release turns them on. Moving the NVMe reads out of Python (one GIL-free C++ call per layer),
deeper read-ahead buffers and limiting how far the host runs ahead of the GPU did not remove it. The cause is
still open; reports from other machines are welcome.

## Development runs

Earlier images of this tree, same protocol (prefill / decode tok/s):

| Image and CPUs | 4K + 256 | 32K + 512 | 131K + 512 | 8K + 1,024 |
|---|---|---|---|---|
| First image, 32 cores, very first start (empty kernel caches) | 783 / 41.3 | 1,155 / 50.3 | 2,076 / 46.1 | 1,074 / 46.4 |
| First image, 32 cores | 1,118 / 50.4 | 2,148 / 47.0 | 2,121 / 45.8 | 1,548 / 47.0 |
| First image, 16 cores | 635 / 45.2 | 1,437 / 43.8 | 982 / 41.0 | 956 / 46.9 |
| + PLE lookups on 4 threads, 32 cores, run 1 | 1,140 / 49.4 | 2,157 / 50.9 | 2,131 / 53.1 | 2,054 / 50.1 |
| + PLE lookups on 4 threads, 32 cores, run 2 | 1,155 / 47.8 | 2,155 / 48.4 | 2,180 / 42.2 | 1,974 / 50.0 |
| + PLE lookups on 4 threads, 16 cores, 2 runs | 1,100–1,126 / 45.6–46.1 | 2,190–2,200 / 45.3–45.6 | 1,472–1,476 / 46.8 | 1,935–1,941 / 46.2–46.6 |
| + PLE lookups on 4 threads, 8 cores | 1,108 / 35.5 | 2,133 / 35.5 | 2,149 / 33.6 | 1,951 / 34.6 |

- The very first start compiles GPU kernels during the first requests; later starts reuse `./jit-cache`.
- The first image let the embedding-table (PLE) lookups run one OpenMP thread per CPU. The threads spun and took
  CPU time from the GPU worker and the expert pool: with 16 cores, long-prompt prefill halved. Four threads
  (`QWEN38_PLE_THREADS`) fixed that and also raised decode on 32 cores.
- The 8-core proxy was not repeated with the release image: it ran the CPU to 89 °C, and a repeat reached the
  90 °C limit of this host and was stopped. Its decode path is the same in the release image.

## Where the time goes

A fast prefill step of 8,192 tokens (32 cores) spends ~2.1 s in attention and other non-expert GPU work, 0.9 s
in cold-expert GEMMs, 0.4 s converting the streamed experts to the kernel layout, and 0.3 s waiting for DMA. In
parallel, the NVMe-tier experts (about 9 GB per step) are read from the PCIe 3.0 SSD in ~2.7 s.

A decode step verifies 4 tokens (about 3 are accepted). In development runs with decode statistics enabled
(`QWEN38_CPU_EXPERTS_STATS=64`), a steady-state step took 33–36 ms of CPU expert work for 750–950 cold experts,
overlapped with ~16 ms of GPU work; the GPU also computes ~22% of the cold experts over PCIe.

## Files

- [`summary.json`](summary.json): every run above, machine-readable.
- [`raw/`](raw/): the `scripts/bench.py` output of each run.
