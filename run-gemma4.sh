#!/bin/bash

set -euo pipefail

# Run Gemma 4 31B-IT on vLLM — optimized for NVIDIA DGX Spark (GB10, 120 GiB, 48 SMs)
# BF16 weights are about 62.5 GiB; with --gpu-memory-utilization 0.85 this leaves
# roughly 40 GiB for KV cache and OS overhead on the GB10's unified memory.
# Requires vLLM >= 0.19.0 with native Gemma 4 support (PR #38826).
#
# Performance tuning (GB10-specific):
#   --enable-prefix-caching             KV-cache reuse for repeated prefixes
#   --enable-chunked-prefill            Prevents prefill OOM, better GPU utilization
#   --max-num-seqs 64                   Reasonable concurrency for 31B dense model
#   --max-num-batched-tokens 8192       Batch saturation without OOM
#   --max-model-len 32768               Gemma 4 supports 128K but 32K is practical on Spark
#   --gpu-memory-utilization 0.85       Leaves headroom for OS and KV cache growth
#   --limit-mm-per-prompt image=0       Text-only: skip vision encoder memory allocation
#
# Tool calling + reasoning:
#   --tool-call-parser gemma4           Native Gemma 4 tool call format (vLLM >= 0.19.0)
#   --reasoning-parser gemma4           Native Gemma 4 thinking mode (<|think|>)
#   --enable-auto-tool-choice           Let vLLM decide when to invoke tools
#
# Ref: https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html

MODEL_NAME="${MODEL_NAME:-google/gemma-4-31B-it}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_gemma4}"
PORT="${PORT:-8004}"
HF_HOME_LOCAL="${HF_HOME_LOCAL:-$HOME/.cache/huggingface}"
VLLM_CACHE_LOCAL="${VLLM_CACHE_LOCAL:-$HOME/.cache/vllm}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEMMA4_GPU_MEMORY_UTILIZATION="${GEMMA4_GPU_MEMORY_UTILIZATION:-0.85}"
GEMMA4_MAX_NUM_SEQS="${GEMMA4_MAX_NUM_SEQS:-64}"
GEMMA4_MAX_NUM_BATCHED_TOKENS="${GEMMA4_MAX_NUM_BATCHED_TOKENS:-8192}"
GEMMA4_MAX_MODEL_LEN="${GEMMA4_MAX_MODEL_LEN:-32768}"

source "${SCRIPT_DIR}/hf-cache.sh"

GEMMA4_HF_HUB_OFFLINE="$(hf_pick_offline_mode "$MODEL_NAME" "$HF_HOME_LOCAL" "${HF_HUB_OFFLINE:-}")"
GEMMA4_TRANSFORMERS_OFFLINE="$(hf_pick_offline_mode "$MODEL_NAME" "$HF_HOME_LOCAL" "${TRANSFORMERS_OFFLINE:-}")"

mkdir -p "$HF_HOME_LOCAL" "$VLLM_CACHE_LOCAL"
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
    -e HF_HUB_OFFLINE="$GEMMA4_HF_HUB_OFFLINE" \
    -e TRANSFORMERS_OFFLINE="$GEMMA4_TRANSFORMERS_OFFLINE" \
    ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
    ${GEMMA4_IMAGE:-vllm-gemma4:latest} \
    vllm serve "$MODEL_NAME" \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization "$GEMMA4_GPU_MEMORY_UTILIZATION" \
        --max-num-seqs "$GEMMA4_MAX_NUM_SEQS" \
        --max-num-batched-tokens "$GEMMA4_MAX_NUM_BATCHED_TOKENS" \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --max-model-len "$GEMMA4_MAX_MODEL_LEN" \
        --limit-mm-per-prompt '{"image":0}' \
        --enable-auto-tool-choice \
        --tool-call-parser gemma4 \
        --reasoning-parser gemma4

echo "Container $CONTAINER_NAME started on port $PORT."
echo "HF_HUB_OFFLINE=$GEMMA4_HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$GEMMA4_TRANSFORMERS_OFFLINE"
echo "Use 'docker logs -f $CONTAINER_NAME' to follow logs."
