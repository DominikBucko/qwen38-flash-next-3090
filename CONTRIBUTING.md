# Contributing: hardware reports, fixes and benchmarks

The most useful contribution right now is **a run on your own machine**, especially a desktop with dual-channel
DDR5 or DDR4, another 24 GB GPU (RTX 3090 Ti, 4090, A5000), or 48/96 GB of RAM. Failed attempts help too.

## Share a run

1. Start the server (`make serve`) and run `make bench` once the log says the server is ready. Run it twice if
   the kernel caches were empty (the first start compiles kernels).
2. Open a [hardware report](https://github.com/DominikBucko/qwen38-flash-next-3090/issues/new?template=hardware-report.yml)
   with the JSON from `results/`, your CPU, RAM (size, speed, channels), GPU, driver, and any settings you changed.
   The startup lines that start with `[qwen38-3090]` show the CPU and memory plan.

Redact private paths, prompts and tokens from logs.

## Report a bug

Use the [bug report form](https://github.com/DominikBucko/qwen38-flash-next-3090/issues/new?template=bug-report.yml).
Check [tuning and troubleshooting](docs/tuning.md) first for out-of-memory problems.

## Pull requests

Keep changes focused and say how you tested them. GPU-free checks:

```bash
make validate
make test
bash -n scripts/*.sh tools/*.sh
```

After changing files in `runtime/vllm-overlay/`, run `make manifest` (the image build verifies the checksums).
For performance changes, compare the same `make bench` shapes before and after, on the same machine, and include
both prefill and decode numbers. For numerical changes, include a correctness check against a reference.
Distinguish measured results from expectations.
