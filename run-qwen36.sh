#!/bin/bash

set -euo pipefail

# Run Qwen3.6-27B-FP8 (dense, FP8) on vLLM — DGX Spark (GB10).
#
# Tracks the official vLLM recipe at https://recipes.vllm.ai/Qwen/Qwen3.6-27B
#   vllm serve Qwen/Qwen3.6-27B-FP8 --max-model-len 262144 --reasoning-parser qwen3
#
# Optional MTP speculative decoding (Qwen3.6 ships native multi-token-prediction
# heads, so no separate draft model is needed):
#   QWEN36_NUM_SPECULATIVE_TOKENS=1 ./run-qwen36.sh
# Set 0 to disable. Useful range is 1–3 per the recipe.

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B-FP8}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_qwen36}"
PORT="${PORT:-8005}"
HF_HOME_LOCAL="${HF_HOME_LOCAL:-$HOME/.cache/huggingface}"
VLLM_CACHE_LOCAL="${VLLM_CACHE_LOCAL:-$HOME/.cache/vllm}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

QWEN36_ATTENTION_BACKEND="${QWEN36_ATTENTION_BACKEND:-FLASHINFER}"
QWEN36_GPU_MEMORY_UTILIZATION="${QWEN36_GPU_MEMORY_UTILIZATION:-0.80}"
QWEN36_MAX_NUM_SEQS="${QWEN36_MAX_NUM_SEQS:-32}"
QWEN36_MAX_NUM_BATCHED_TOKENS="${QWEN36_MAX_NUM_BATCHED_TOKENS:-8192}"
QWEN36_BLOCK_SIZE="${QWEN36_BLOCK_SIZE:-16}"
QWEN36_MAX_MODEL_LEN="${QWEN36_MAX_MODEL_LEN:-65536}"
QWEN36_NUM_SPECULATIVE_TOKENS="${QWEN36_NUM_SPECULATIVE_TOKENS:-0}"
QWEN36_KV_CACHE_DTYPE="${QWEN36_KV_CACHE_DTYPE:-auto}"

source "${SCRIPT_DIR}/hf-cache.sh"

QWEN36_HF_HUB_OFFLINE="$(hf_pick_offline_mode "$MODEL_NAME" "$HF_HOME_LOCAL" "${HF_HUB_OFFLINE:-}")"
QWEN36_TRANSFORMERS_OFFLINE="$(hf_pick_offline_mode "$MODEL_NAME" "$HF_HOME_LOCAL" "${TRANSFORMERS_OFFLINE:-}")"

mkdir -p "$HF_HOME_LOCAL" "$VLLM_CACHE_LOCAL"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

vllm_extra_args=()
if [ "${QWEN36_NUM_SPECULATIVE_TOKENS:-0}" != 0 ]; then
  vllm_extra_args+=(
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${QWEN36_NUM_SPECULATIVE_TOKENS}}"
  )
fi
if [ "${QWEN36_KV_CACHE_DTYPE:-auto}" != auto ]; then
  vllm_extra_args+=(--kv-cache-dtype "$QWEN36_KV_CACHE_DTYPE")
fi

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
    -e VLLM_ATTENTION_BACKEND="$QWEN36_ATTENTION_BACKEND" \
    -e HF_HUB_OFFLINE="$QWEN36_HF_HUB_OFFLINE" \
    -e TRANSFORMERS_OFFLINE="$QWEN36_TRANSFORMERS_OFFLINE" \
    ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
    scitrera/dgx-spark-vllm:0.17.0-t5 \
    vllm serve "$MODEL_NAME" \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization "$QWEN36_GPU_MEMORY_UTILIZATION" \
        --max-num-seqs "$QWEN36_MAX_NUM_SEQS" \
        --max-num-batched-tokens "$QWEN36_MAX_NUM_BATCHED_TOKENS" \
        --block-size "$QWEN36_BLOCK_SIZE" \
        --enable-prefix-caching \
        --max-model-len "$QWEN36_MAX_MODEL_LEN" \
        --reasoning-parser qwen3 \
        "${vllm_extra_args[@]}"

echo "Container $CONTAINER_NAME started on port $PORT."
echo "HF_HUB_OFFLINE=$QWEN36_HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$QWEN36_TRANSFORMERS_OFFLINE"
echo "QWEN36_NUM_SPECULATIVE_TOKENS=$QWEN36_NUM_SPECULATIVE_TOKENS (0 = MTP disabled)"
echo "Use 'docker logs -f $CONTAINER_NAME' to follow logs."
