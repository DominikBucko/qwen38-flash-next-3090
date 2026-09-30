#!/usr/bin/env bash
# Check this host against the single-GPU profile before the first start. Errors stop `make serve`.
set -euo pipefail

fail=0
warn() { printf 'WARN:  %s\n' "$*" >&2; }
error() { printf 'ERROR: %s\n' "$*" >&2; fail=1; }
ok() { printf 'ok:    %s\n' "$*"; }

[[ "$(uname -s)" == Linux ]] || error "serving requires Linux"
[[ "$(uname -m)" == x86_64 ]] || error "the CPU expert kernel needs an x86-64 CPU"
command -v docker >/dev/null || error "docker is not installed"
command -v nvidia-smi >/dev/null || error "nvidia-smi is not installed (NVIDIA driver)"
if command -v docker >/dev/null && ! docker info >/dev/null 2>&1; then
  error "the Docker daemon is not reachable for this user (add the user to the docker group, or run with sudo)"
fi
if command -v docker >/dev/null && docker info >/dev/null 2>&1 && ! docker info 2>/dev/null | grep -qi nvidia; then
  warn "docker info does not mention the NVIDIA runtime; install the NVIDIA Container Toolkit if --gpus fails"
fi

# CPU: the cold-expert kernel uses AVX2, FMA and F16C.
if [[ -r /proc/cpuinfo ]]; then
  flags=$(grep -m1 '^flags' /proc/cpuinfo)
  for f in avx2 fma f16c; do
    [[ " $flags " == *" $f "* ]] || error "CPU lacks $f"
  done
  ok "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//'), $(nproc) threads"
fi

# NVIDIA driver: the image uses CUDA 13.0 (driver 580 or newer).
if command -v nvidia-smi >/dev/null; then
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  if [[ -n "$driver" ]]; then
    (( ${driver%%.*} >= 580 )) && ok "NVIDIA driver $driver" || error "NVIDIA driver $driver is too old: CUDA 13.0 needs 580 or newer"
  fi
fi

# GPU: one 24 GB card; the default profile peaks at about 23.4 GB.
gpu=${GPU_DEVICE:-0}
if command -v nvidia-smi >/dev/null; then
  line=$(nvidia-smi -i "$gpu" --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null || true)
  if [[ -z "$line" ]]; then
    error "GPU $gpu not found (set GPU_DEVICE)"
  else
    IFS=',' read -r name total used <<< "$line"
    total=${total// /}; used=${used// /}
    ok "GPU $gpu: ${name# } (${total} MiB, ${used} MiB in use)"
    (( total >= 23000 )) || error "GPU $gpu has ${total} MiB; the profile needs a 24 GB card"
    (( used <= 1000 )) || warn "${used} MiB of GPU $gpu is already in use (display or other programs); the profile peaks at ~23,400 MiB, see docs/tuning.md if it runs out of GPU memory"
  fi
fi

# RAM: 64 GB is the tested size. The expert arena is sized from what the container may use.
if [[ -r /proc/meminfo ]]; then
  mem_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
  avail_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  mem_gib=$(( mem_kib / 1048576 )); avail_gib=$(( avail_kib / 1048576 ))
  ok "RAM: ${mem_gib} GiB total, ${avail_gib} GiB available"
  (( mem_kib >= 41943040 )) || error "at least 48 GB of RAM is needed (64 GB tested)"
  (( mem_kib >= 60817408 )) || warn "less than 64 GB of RAM: more cold experts stay on NVMe and decode is slower (untested)"
  installed_gib=$(( (mem_kib + 8388607) / 8388608 * 8 ))
  limit_gib=$(( installed_gib - ${HOST_RESERVE_GIB:-8} ))
  ok "container memory limit (auto): ${limit_gib} GiB of ~${installed_gib} GB installed"
  (( avail_kib / 1048576 + 1 >= limit_gib )) || warn "only ${avail_gib} GiB is available now for a ${limit_gib} GiB container; close other programs (or set MEMORY_LIMIT lower) to keep the page cache"
fi

# Transparent huge pages: the expert arena asks for them (madvise is enough).
thp=/sys/kernel/mm/transparent_hugepage/enabled
if [[ -r $thp ]]; then
  grep -q '\[never\]' "$thp" && warn "transparent huge pages are disabled; decode is a little slower without them"
fi
kernel=$(uname -r)
IFS=. read -r kmaj kmin _ <<< "$kernel"
if (( kmaj < 5 || (kmaj == 5 && kmin < 14) )); then
  warn "kernel $kernel: 5.14+ recommended (MADV_POPULATE_WRITE), 6.1+ for huge-page collapse"
fi

# Checkpoint: read at runtime (PLE rows and NVMe-tier experts), so it belongs on an NVMe SSD.
if [[ -n "${MODEL_DIR:-}" ]]; then
  if [[ ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
    error "MODEL_DIR=$MODEL_DIR is not the downloaded checkpoint"
  else
    [[ -f "$MODEL_DIR/runtime/mtp-int4-g32/model.safetensors.index.json" ]] || error "MTP draft missing under $MODEL_DIR/runtime/mtp-int4-g32"
    src=$(df --output=source "$MODEL_DIR" 2>/dev/null | tail -1)
    dev=$(basename "$(readlink -f "$src")")
    parent=$(lsblk -no pkname "/dev/$dev" 2>/dev/null | head -1 || true)
    disk=${parent:-$dev}
    if [[ "$disk" == nvme* ]]; then
      ok "checkpoint on NVMe ($disk)"
    elif [[ "$(cat "/sys/block/$disk/queue/rotational" 2>/dev/null || echo 0)" == 1 ]]; then
      error "the checkpoint is on a spinning disk ($disk); the runtime reads it continuously, use an NVMe SSD"
    else
      warn "the checkpoint is not on an NVMe device ($disk); prefill and decode read it at runtime"
    fi
  fi
else
  warn "MODEL_DIR is not set; skipped the checkpoint checks"
fi

(( fail == 0 )) || exit 1
printf 'preflight passed\n'
