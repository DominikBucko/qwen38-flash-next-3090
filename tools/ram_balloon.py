"""Memory balloon: lock (MemTotal - TARGET_GIB) of RAM so the host behaves like a TARGET_GIB machine.

Run in a container with --ulimit memlock=-1 (see ram-balloon.sh). Holds the memory until SIGTERM.
"""
import ctypes
import mmap
import os
import signal
import sys
import time

target_gib = float(os.environ.get("TARGET_GIB", "64"))
with open("/proc/meminfo") as fh:
    total_kib = next(int(line.split()[1]) for line in fh if line.startswith("MemTotal:"))
size = int(total_kib * 1024 - target_gib * 2**30)
size -= size % mmap.PAGESIZE
if size <= 0:
    sys.exit(f"MemTotal {total_kib / 2**20:.1f} GiB is not above the target {target_gib} GiB")
buf = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
# Huge pages: the balloon then occupies whole 2 MB blocks instead of scattering 4K pages over all of RAM,
# so the memory left for the server stays usable for its own huge-page arena.
buf.madvise(mmap.MADV_HUGEPAGE)
libc = ctypes.CDLL("libc.so.6", use_errno=True)
addr = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(buf)))
t0 = time.time()
step = 1 << 30
for off in range(0, size, step):
    n = min(step, size - off)
    if libc.mlock(ctypes.c_void_p(addr.value + off), ctypes.c_size_t(n)) != 0:
        sys.exit(f"mlock failed at {off / 2**30:.1f} GiB: errno {ctypes.get_errno()}")
print(f"balloon: locked {size / 2**30:.2f} GiB of {total_kib / 2**20:.2f} GiB in {time.time() - t0:.1f} s "
      f"-> {target_gib} GiB left", flush=True)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
while True:
    time.sleep(3600)
