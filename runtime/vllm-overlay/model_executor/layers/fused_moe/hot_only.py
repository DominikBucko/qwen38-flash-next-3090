# SPDX-License-Identifier: Apache-2.0
"""Single-GPU hot-only expert placement (QWEN38_HOT_ONLY=<experts per layer>).

The GPU owns only each target layer's hot experts, exactly like one expert-parallel rank that holds a subset: the
expert map sends the hot global ids to local slots 0..H-1 and every other id to -1, so the loader skips cold
experts and the kernels skip their tokens. Cold experts are computed from the host copy instead (CPU for decode,
streamed to the GPU for prefill; see compressed_tensors_moe/expert_store.py). The hot set is the first H ids of
the static rankings file, per layer. The MTP draft layer is not affected.
"""
import json
import os
import re

import torch

_RANKINGS = None


def size() -> int:
    return int(os.environ.get("QWEN38_HOT_ONLY", "0"))


def layer_index(layer_name: str) -> int | None:
    if "language_model.model.layers." not in layer_name:
        return None
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    return int(match.group(1)) if match else None


def _rankings() -> dict:
    global _RANKINGS
    if _RANKINGS is None:
        path = os.getenv("VLLM_WNA16_STATIC_HOT_CACHE_FILE") or os.path.join(os.getcwd(), "static_hot_cache_rankings.json")
        with open(path) as fh:
            _RANKINGS = json.load(fh)
    return _RANKINGS


def hot_ids(layer: int, global_num_experts: int) -> list[int]:
    count = size()
    ids, seen = [], set()
    for raw in _rankings().get(str(layer), []):
        g = int(raw)
        if 0 <= g < global_num_experts and g not in seen:
            ids.append(g)
            seen.add(g)
            if len(ids) == count:
                break
    if len(ids) != count:
        raise RuntimeError(f"hot-only: layer {layer} ranks {len(ids)} experts, need {count}")
    return ids


def apply_placement(manager, layer_name: str, global_num_experts: int) -> None:
    """Turn a no-EP ExpertMapManager into a hot-only one for target MoE layers."""
    if size() <= 0:
        return
    layer = layer_index(layer_name)
    if layer is None:
        return
    if manager.expert_map is not None or manager.num_fused_shared_experts:
        raise RuntimeError("hot-only placement requires a single GPU without expert parallelism")
    ids = hot_ids(layer, global_num_experts)
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    expert_map[torch.tensor(ids, dtype=torch.long)] = torch.arange(len(ids), dtype=torch.int32)
    manager._local_num_experts = len(ids)
    manager._expert_map = expert_map
    manager._qwen38_hot_ids = ids
