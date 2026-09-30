"""tail_read (cpu_moe.cpp) against plain file reads: random parts of a temporary file, several thread counts.
Runs inside the image (needs the built extension, no GPU):
  docker run --rm -v "$PWD/tests:/tests:ro" --entrypoint python3 IMAGE /tests/image/test_tail_read.py
"""
import os
import random
import tempfile

import torch

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.cpu_experts import extension

ext = extension()
rng = random.Random(0)
size = 64 << 20
data = os.urandom(size)
with tempfile.NamedTemporaryFile() as fh:
    fh.write(data)
    fh.flush()
    fd = os.open(fh.name, os.O_RDONLY)
    for threads in (1, 3, 8):
        out = torch.zeros(96 << 20, dtype=torch.uint8)
        rows, dst = [], 0
        for _ in range(500):
            n = rng.randrange(1, 300_000)
            off = rng.randrange(0, size - n)
            rows.append((fd, off, n, dst, rng.randrange(2)))
            dst += n + rng.randrange(0, 64)
        got = ext.tail_read(out.data_ptr(), torch.tensor(rows, dtype=torch.int64), threads)
        assert got == sum(r[2] for r in rows), got
        mem = out.numpy().tobytes()
        for _, off, n, d, _ in rows:
            assert mem[d:d + n] == data[off:off + n], (off, n, d)
        print(f"threads={threads}: {len(rows)} parts, {got / 2**20:.1f} MiB, byte-exact")
    try:
        ext.tail_read(out.data_ptr(), torch.tensor([(fd, size - 10, 100, 0, 0)], dtype=torch.int64), 2)
        raise SystemExit("short read not detected")
    except RuntimeError as err:
        print("short read detected:", str(err).splitlines()[0])
    os.close(fd)
print("tail_read OK")
