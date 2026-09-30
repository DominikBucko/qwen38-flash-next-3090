#!/usr/bin/env bash
# Start the server container: one GPU, a memory limit that leaves HOST_RESERVE_GIB for the OS and desktop, and
# persistent kernel caches. The OpenAI-compatible endpoint listens on 127.0.0.1:$PORT.
set -euo pipefail

image=${IMAGE:-qwen38-flash-next-3090:local}
model_dir=${MODEL_DIR:?Set MODEL_DIR to the downloaded checkpoint directory}
port=${PORT:-8000}
gpu=${GPU_DEVICE:-0}

model_dir=$(cd -- "$model_dir" && pwd -P)
[[ -f "$model_dir/model.safetensors.index.json" ]] || {
  echo "MODEL_DIR is not a model checkpoint: $model_dir" >&2
  exit 2
}

# Memory: the whole serving stack (expert arena, processes, page cache) stays inside this limit, and the runtime
# sizes its expert arena from it. "auto": installed RAM (MemTotal rounded up to a multiple of 8 GiB) minus
# HOST_RESERVE_GIB (default 8) for the OS, i.e. 56 GiB on a 64 GB machine.
memory_limit=${MEMORY_LIMIT:-auto}
if [[ "$memory_limit" == auto ]]; then
  mem_total_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
  installed_gib=$(( (mem_total_kib + 8388607) / 8388608 * 8 ))
  memory_limit="$(( installed_gib - ${HOST_RESERVE_GIB:-8} ))g"
fi
cpuset_args=()
[[ -n "${CPUSET:-}" ]] && cpuset_args=(--cpuset-cpus "$CPUSET")

# Humming and Triton compile kernels on first use; keeping the caches makes later starts faster.
jit_cache_dir=${JIT_CACHE_DIR:-./jit-cache}
mkdir -p "$jit_cache_dir/humming" "$jit_cache_dir/triton"
jit_cache_dir=$(cd -- "$jit_cache_dir" && pwd -P)

docker_env=(-e "PORT=$port")
for name in SERVED_MODEL_NAME MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS MAX_PARALLEL_LOADING_WORKERS \
  KV_CACHE_DTYPE KV_CACHE_MEMORY_BYTES MTP_DEPTH PYTORCH_CUDA_ALLOC_CONF \
  VLLM_PREFIX_CACHE_RETENTION_INTERVAL VLLM_MTP_DRAFT_VOCAB_RANGES VLLM_PLE_OFFLOAD_READY_TIMEOUT \
  VLLM_QSA_EXACT_TOPK VLLM_API_KEY; do
  if declare -p "$name" &>/dev/null; then
    docker_env+=(-e "$name=${!name}")
  fi
done
# Every QWEN38_* runtime setting in the environment (for example from .env) is passed through.
while IFS= read -r name; do
  docker_env+=(-e "$name=${!name}")
done < <(compgen -e | grep -E '^QWEN38_[A-Z0-9_]+$' || true)

exec docker run --rm \
  --name "${CONTAINER_NAME:-qwen38-flash-next-3090}" \
  --gpus "device=$gpu" \
  --ipc host \
  --cap-add SYS_PTRACE \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --memory "$memory_limit" \
  --memory-swap "$memory_limit" \
  ${cpuset_args[@]+"${cpuset_args[@]}"} \
  -p "127.0.0.1:$port:$port" \
  "${docker_env[@]}" \
  -v "$model_dir:/model:ro" \
  -v "$jit_cache_dir/humming:/root/.humming" \
  -v "$jit_cache_dir/triton:/root/.triton" \
  "$image" /model
