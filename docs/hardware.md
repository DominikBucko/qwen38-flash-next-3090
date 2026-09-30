# Hardware: requirements, the benchmark host, and what to expect

## Requirements

| Part | Needed | Notes |
|---|---|---|
| GPU | One NVIDIA card with 24 GB | RTX 3090 tested. RTX 3090 Ti, RTX 4090, RTX A5000 and other 24 GB cards should work but are untested. 16 GB cards do not fit the default profile. |
| System RAM | 64 GB | Tested size. 96 GB or more keeps every expert in RAM; 48 GB should start but decode will read more from NVMe (untested). |
| CPU | x86-64 with AVX2, FMA and F16C | Any Ryzen, or Intel Core from Haswell (2013) on. Decode speed depends on memory bandwidth (see below). |
| Storage | NVMe SSD, ~130 GB free | The checkpoint is read at runtime (PLE rows, least-used experts). A SATA SSD is slower; a hard disk is not usable. |
| Software | Linux, Docker, NVIDIA Container Toolkit, NVIDIA driver 580+ | The image uses CUDA 13.0. |

`make preflight` checks all of this on your machine.

## Benchmark host

| Part | Configuration |
|---|---|
| GPU | 1× RTX 3090 24 GB, PCIe 4.0 ×16 |
| CPU | AMD Threadripper PRO 5975WX, 32 cores / 64 threads, 4 CCDs |
| RAM | 8×16 GB DDR4-3200, limited to **64 GiB** with a memory balloon ([`tools/ram-balloon.sh`](../tools/ram-balloon.sh)); the server container was capped at **56 GiB** |
| NVMe | Samsung PM981a 1 TB (PCIe 3.0 ×4), also the system disk |
| Software | Linux 7.0, NVIDIA driver 595.84, Docker |

The balloon locks the RAM above 64 GiB in a separate process, so the page cache, the expert arena and the
serving processes compete for 64 GiB exactly as on a 64 GB machine. It was started after the server had
populated its expert arena, because a freshly booted 64 GB machine hands out unfragmented huge pages.

## What to expect on a desktop

Prefill is GPU- and PCIe-bound, so it should be similar on any machine with a PCIe 4.0 ×16 slot (PCIe 3.0
halves the expert streaming bandwidth; expect slower long prompts).

Decode is bound by how fast the CPU reads the cold experts: about 2 GB per decode step. During serving, the
benchmark host's 24-thread pool reads them at about 60 GB/s (85 GB/s in an isolated test). Its eight DDR4
channels could deliver far more; what limits it is the link between each CCD and memory. For comparison, the
theoretical bandwidth of dual-channel DDR5-6000 is 96 GB/s, DDR5-4800 77 GB/s and DDR4-3200 51 GB/s; real
reads reach a good part of that, depending on the CPU.

So a DDR5 desktop should be in the same range as the benchmark host's pool, and a DDR4 desktop clearly lower. The [desktop CPU proxy runs](../benchmarks/2026-09-30/README.md#desktop-cpu-proxies) restrict
the server to 16 and 8 cores of the benchmark CPU to show how decode scales with fewer cores and less
bandwidth. They are proxies, not desktop measurements. Please [share your numbers](https://github.com/DominikBucko/qwen38-flash-next-3090/issues/new?template=hardware-report.yml).

Tips for desktops:

- Populate both memory channels (two or four DIMMs) and enable EXPO/XMP.
- On two-CCD Ryzen parts (7900X, 7950X, 9900X, 9950X), the CPU pool spans both CCDs automatically.
- Keep other memory-heavy programs closed while generating: they share the same bandwidth.
- If the 3090 also drives your display, see [GPU memory](tuning.md#gpu-memory).

## More RAM or another GPU

- **96–128 GB RAM, one GPU**: the whole expert set fits in RAM; decode never reads experts from NVMe. The
  default `auto` settings pick this up.
- **Two RTX 3090s and 64 GB**: use the [64 GB profile](https://github.com/DominikBucko/qwen38-flash-next-2x3090#new-64-gb-ram-profile)
  of [qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090). It applies this
  runtime's design to both cards, each owning half of the experts: 3,410 tok/s prefill and 84 tok/s decode on a
  131K prompt.
- **Two RTX 3090s and 128 GB**: the same repository's 128 GB profiles keep all cold-expert work on the GPUs:
  up to ~4,200 tok/s prefill (prefill profile) and ~110 tok/s decode (agent profile).
