#!/usr/bin/env bash
# Container entrypoint: serve Qwen3.8-Flash-Next on one 24 GB GPU. Hot experts run on the GPU; the CPU computes
# the cold experts during decode, and prefill streams them to the GPU. Usage: qwen38-serve [MODEL_DIR [MTP_DIR]]
set -euo pipefail

profile=${QWEN38_PROFILE:-/opt/qwen38-3090/configs/3090-64gb-128k.env}
[[ -f "$profile" ]] || { echo "missing runtime profile: $profile" >&2; exit 2; }
# shellcheck source=/dev/null
source "$profile"

model=${1:-/model}
mtp_model=${2:-"$model/runtime/mtp-int4-g32"}
rankings=/opt/qwen38-3090/configs/static_hot_cache_rankings.json

[[ -f "$model/model.safetensors.index.json" ]] || { echo "model checkpoint not found at $model" >&2; exit 2; }
[[ -f "$mtp_model/model.safetensors.index.json" ]] || { echo "MTP draft not found at $mtp_model" >&2; exit 2; }
[[ -f "$rankings" ]] || { echo "missing expert rankings: $rankings" >&2; exit 2; }

# The CPU decode path takes one request with up to 4 tokens per step (MTP depth <= 3).
[[ "$MAX_NUM_SEQS" == 1 ]] || { echo "MAX_NUM_SEQS must be 1 on the single-GPU runtime" >&2; exit 2; }
[[ "$MTP_DEPTH" =~ ^[0-3]$ ]] || { echo "MTP_DEPTH must be 0-3 on the single-GPU runtime" >&2; exit 2; }
if [[ "$KV_CACHE_DTYPE" == int8_per_token_head || "$KV_CACHE_DTYPE" == fp8_per_token_head ]]; then
  (( VLLM_PREFIX_CACHE_RETENTION_INTERVAL % 3200 == 0 )) || {
    echo "VLLM_PREFIX_CACHE_RETENTION_INTERVAL must be a multiple of 3200 with $KV_CACHE_DTYPE" >&2; exit 2; }
fi
case "$QWEN38_ASYNC_SCHEDULING" in
  0) async_scheduling_arg=--no-async-scheduling ;;
  1) async_scheduling_arg=--async-scheduling ;;
  *) echo "QWEN38_ASYNC_SCHEDULING must be 0 or 1" >&2; exit 2 ;;
esac
if (( MTP_DEPTH > 0 )); then
  spec_args=(--speculative-config
    "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_DEPTH,\"use_local_argmax_reduction\":true,\"model\":\"$mtp_model\"}")
else
  spec_args=()
fi

# Single-GPU mode (fixed): hot-only expert placement, CPU cold experts, PLE table read from the checkpoint on NVMe.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export QWEN38_CPU_EXPERTS=1 QWEN38_CPU_EXPERTS_MODEL="$model"
export QWEN38_PLE_MMAP=1 QWEN38_PLE_MMAP_DIR="$model" QWEN38_PLE_PREFAULT=0
export QWEN38_STREAM_STAGE=0 QWEN38_STAGE_OVERLAP=0 CPU_OFFLOAD_GB=0
export VLLM_PLE_CPU_OFFLOAD=1 VLLM_FORCE_DYNAMIC_SPEC_SCHEDULING=1
export VLLM_WNA16_DYNAMIC_LRU=0 VLLM_WNA16_MIXED_VMM_HOT_CACHE=0 VLLM_WNA16_STATIC_HOT_CACHE_SIZE=0
export VLLM_WNA16_STATIC_HOT_CACHE_MAX_TOKENS=16 VLLM_WNA16_STATIC_HOT_CACHE_FILE="$rankings"

# Host plan: resolve the "auto" CPU sets and arena size once, so every serving process uses the same values.
host_py=$(python3 -c 'import importlib.util, os; print(os.path.join(os.path.dirname(importlib.util.find_spec("vllm").origin), "qwen38_host.py"))')
while IFS='=' read -r key value; do
  export "$key=$value"
done < <(python3 "$host_py" --env)
python3 "$host_py" | sed 's/^/[qwen38-3090] /'

exec vllm serve "$model" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 1 \
  --distributed-executor-backend mp \
  --moe-backend humming \
  --dtype bfloat16 \
  --language-model-only \
  --load-format safetensors \
  --safetensors-load-strategy lazy \
  --max-parallel-loading-workers "$MAX_PARALLEL_LOADING_WORKERS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  "$async_scheduling_arg" \
  --disable-custom-all-reduce \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  ${spec_args[@]+"${spec_args[@]}"}
