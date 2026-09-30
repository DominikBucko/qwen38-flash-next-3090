# Qwen3.8-Flash-Next on one RTX 3090 (24 GB) + 64 GB RAM.
# Base: the v0.4.0 release image of github.com/DominikBucko/qwen38-flash-next-2x3090 (the pinned vLLM build, Humming
# kernels and the Qwen3.8 runtime overlay). This image adds the single-GPU overlay on top. No model weights inside.
FROM ghcr.io/dominikbucko/qwen38-flash-next-2x3090@sha256:139e796cf3375c26f08a005762bbbf2d1d5495d9fe9e7b3c858dbe506550dd33

COPY runtime/install_overlay.py /opt/qwen38-3090/runtime/install_overlay.py
COPY runtime/vllm-overlay /opt/qwen38-3090/runtime/vllm-overlay
RUN python3 /opt/qwen38-3090/runtime/install_overlay.py /opt/qwen38-3090/runtime/vllm-overlay

# CPU cold-expert kernel and its GPU bridge, compiled once here (x86-64 AVX2/FMA/F16C; sm_86 plus PTX) so the
# server never compiles it at startup.
ENV QWEN38_CPU_EXPERTS_BUILD=/opt/qwen38-3090/cpu_experts
RUN MAX_JOBS=2 python3 -c 'from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.cpu_experts import build_extension; build_extension(verbose=True)'

COPY configs/3090-64gb-128k.env /opt/qwen38-3090/configs/3090-64gb-128k.env
COPY configs/static_hot_cache_rankings.json /opt/qwen38-3090/configs/static_hot_cache_rankings.json
COPY scripts/serve-container.sh /usr/local/bin/qwen38-serve
RUN chmod 0755 /usr/local/bin/qwen38-serve

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/qwen38-serve"]
