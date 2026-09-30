# Third-party notices

This repository does not contain model weights.

- The runtime overlay is derived from vLLM and from the
  [qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090) overlay, and is distributed
  under the Apache License 2.0; see [`LICENSE`](LICENSE).
- The model (Qwen/Qwen3.8-Flash-Next and the assembled checkpoint
  [albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE)) is
  governed by the Qwen Community License supplied with the upstream checkpoint. Intel's AutoRound checkpoint and
  RadixArk's NVFP4 checkpoint remain governed by their own model cards, notices and inherited Qwen terms.
- The container image builds on the qwen38-flash-next-2x3090 release image, which is based on
  `vllm/vllm-openai` and includes PyTorch, CUDA and the Humming kernels under their own licenses.

Review the current upstream terms before using or redistributing the model.
