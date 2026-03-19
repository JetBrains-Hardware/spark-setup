#!/bin/bash

set -exu

# Run Qwen3-Coder-Next on vLLM 0.16 — optimized for NVIDIA DGX Spark (GB10, 120 GiB, 48 SMs)
# Uses FP8 quantization. Model weights ~78 GiB, ~3.9B active params (MoE).
# Starts in daemon mode; container auto-restarts on reboot.
#
# Performance tuning (GB10-specific):
#   VLLM_USE_DEEP_GEMM=0              Avoid experimental GEMM that bloats peak memory
#   VLLM_ALLOW_LONG_MAX_MODEL_LEN=1   Required for 256K context
#   VLLM_USE_V1=1                     Use vLLM V1 engine for better memory management
#   VLLM_ATTENTION_BACKEND=FLASHINFER GQA-optimized attention for Qwen3-Coder-Next
#
#   NOTE: VLLM_USE_FLASHINFER_MOE_FP8=1 does NOT work on GB10 (SM121) with
#   Qwen3-Coder-Next-FP8's block quantization [128,128]. TRTLLM requires SM100,
#   CUTLASS doesn't support the quant scheme. Triton backend is used instead.
#
#   --enable-prefix-caching            Mamba prefix caching (align mode, vLLM 0.15+)
#   --enable-chunked-prefill           Better GPU utilization, prevents prefill OOM
#   --max-num-seqs 64                  Increased from 16 for higher throughput
#   --max-num-batched-tokens 8192      More tokens per batch for saturation
#   --block-size 128                   MoE-friendly block size
#   --max-model-len 200000             Reduced from 256K to fit the Spark reliably
#   --gpu-memory-utilization 0.75      Leaves ~30 GiB for FLUX image model + overhead
#   --disable-log-requests             Reduce logging overhead
#
# Ref: https://forums.developer.nvidia.com/t/357820
# Ref: https://docs.vllm.ai/en/latest/configuration/engine_args/

QWEN_HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
QWEN_TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Ensure persistent host directories exist before bind-mounting them.
mkdir -p ~/.cache/vllm

# Stop any existing container with the same name
docker rm -f vllm_qwen_code >/dev/null 2>&1 || true

docker run -d \
    --gpus all \
    --name vllm_qwen_code \
    --restart unless-stopped \
    -p 8000:8000 \
    --ipc=host \
    --shm-size=32g \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/vllm:/root/.cache/vllm \
    -e VLLM_USE_DEEP_GEMM=0 \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e VLLM_USE_V1=1 \
    -e VLLM_ATTENTION_BACKEND=FLASHINFER \
    -e HF_HUB_OFFLINE="$QWEN_HF_HUB_OFFLINE" \
    -e TRANSFORMERS_OFFLINE="$QWEN_TRANSFORMERS_OFFLINE" \
    ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
    scitrera/dgx-spark-vllm:0.16.0-t5 \
    vllm serve Qwen/Qwen3-Coder-Next-FP8 \
        --port 8000 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.75 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 8192 \
        --block-size 128 \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --max-model-len 200000 \
        --disable-log-requests \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder

echo "Container vllm_qwen_code started. Use 'docker logs -f vllm_qwen_code' to follow logs."
