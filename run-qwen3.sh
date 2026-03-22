#!/bin/bash

set -euo pipefail

# Run Qwen3-Coder-Next on vLLM 0.16 — optimized for NVIDIA DGX Spark (GB10, 120 GiB, 48 SMs)
# Uses FP8 quantization. Model weights are about 78 GiB, with 80B total parameters and 3B active parameters.
# Starts in daemon mode; container auto-restarts on reboot.
#
# Performance tuning (GB10-specific):
#   VLLM_USE_DEEP_GEMM=0              Avoid experimental GEMM that bloats peak memory
#   VLLM_ALLOW_LONG_MAX_MODEL_LEN=1   Needed for the 200000-token cap below
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
#   --gpu-memory-utilization 0.75      Leaves about 30 GiB of Spark headroom for OS and KV cache growth
#   --disable-log-requests             Reduce logging overhead
#
# Ref: https://forums.developer.nvidia.com/t/357820
# Ref: https://docs.vllm.ai/en/latest/configuration/engine_args/

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Coder-Next-FP8}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_qwen_code}"
PORT="${PORT:-8000}"
HF_HOME_LOCAL="${HF_HOME_LOCAL:-$HOME/.cache/huggingface}"
VLLM_CACHE_LOCAL="${VLLM_CACHE_LOCAL:-$HOME/.cache/vllm}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_ATTENTION_BACKEND="${QWEN_ATTENTION_BACKEND:-FLASHINFER}"
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.75}"
QWEN_MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-64}"
QWEN_MAX_NUM_BATCHED_TOKENS="${QWEN_MAX_NUM_BATCHED_TOKENS:-8192}"
QWEN_BLOCK_SIZE="${QWEN_BLOCK_SIZE:-128}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-200000}"

# Reuse the same cache probe logic as the Nemotron launcher.
source "${SCRIPT_DIR}/hf-cache.sh"

QWEN_HF_HUB_OFFLINE="$(hf_pick_offline_mode "$MODEL_NAME" "$HF_HOME_LOCAL" "${HF_HUB_OFFLINE:-}")"
QWEN_TRANSFORMERS_OFFLINE="$(hf_pick_offline_mode "$MODEL_NAME" "$HF_HOME_LOCAL" "${TRANSFORMERS_OFFLINE:-}")"

# Ensure persistent host directories exist before bind-mounting them.
mkdir -p "$HF_HOME_LOCAL" "$VLLM_CACHE_LOCAL"

# Stop any existing container with the same name
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
    --gpus all \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    -p "$PORT:$PORT" \
    --ipc=host \
    --shm-size=32g \
    -v "$HF_HOME_LOCAL:/root/.cache/huggingface" \
    -v "$VLLM_CACHE_LOCAL:/root/.cache/vllm" \
    -e VLLM_USE_DEEP_GEMM=0 \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e VLLM_USE_V1=1 \
    -e VLLM_ATTENTION_BACKEND="$QWEN_ATTENTION_BACKEND" \
    -e HF_HUB_OFFLINE="$QWEN_HF_HUB_OFFLINE" \
    -e TRANSFORMERS_OFFLINE="$QWEN_TRANSFORMERS_OFFLINE" \
    ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
    scitrera/dgx-spark-vllm:0.17.0-t5 \
    vllm serve "$MODEL_NAME" \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization "$QWEN_GPU_MEMORY_UTILIZATION" \
        --max-num-seqs "$QWEN_MAX_NUM_SEQS" \
        --max-num-batched-tokens "$QWEN_MAX_NUM_BATCHED_TOKENS" \
        --block-size "$QWEN_BLOCK_SIZE" \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --max-model-len "$QWEN_MAX_MODEL_LEN" \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder

echo "Container $CONTAINER_NAME started on port $PORT."
echo "HF_HUB_OFFLINE=$QWEN_HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$QWEN_TRANSFORMERS_OFFLINE"
echo "Use 'docker logs -f $CONTAINER_NAME' to follow logs."
