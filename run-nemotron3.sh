#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="${MODEL_NAME:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_nemotron3}"
PORT="${PORT:-8003}"
HF_HOME_LOCAL="${HF_HOME_LOCAL:-$HOME/.cache/huggingface}"
VLLM_CACHE_LOCAL="${VLLM_CACHE_LOCAL:-$HOME/.cache/vllm}"
PARSER_FILE="${SCRIPT_DIR}/super_v3_reasoning_parser.py"
NEMOTRON_GPU_MEMORY_UTILIZATION="${NEMOTRON_GPU_MEMORY_UTILIZATION:-0.90}"
NEMOTRON_ATTENTION_BACKEND="${NEMOTRON_ATTENTION_BACKEND:-TRITON_ATTN}"
NEMOTRON_BLOCK_SIZE="${NEMOTRON_BLOCK_SIZE:-64}"
NEMOTRON_MAX_NUM_BATCHED_TOKENS="${NEMOTRON_MAX_NUM_BATCHED_TOKENS:-16384}"
NEMOTRON_MAX_NUM_SEQS="${NEMOTRON_MAX_NUM_SEQS:-512}"
NEMOTRON_MAX_MODEL_LEN="${NEMOTRON_MAX_MODEL_LEN:-262144}"
source "${SCRIPT_DIR}/hf-cache.sh"

NEMOTRON_HF_HUB_OFFLINE="$(hf_pick_offline_mode "$MODEL_NAME" "$HF_HOME_LOCAL" "${HF_HUB_OFFLINE:-}")"
NEMOTRON_TRANSFORMERS_OFFLINE="$(hf_pick_offline_mode "$MODEL_NAME" "$HF_HOME_LOCAL" "${TRANSFORMERS_OFFLINE:-}")"

if [ ! -f "$PARSER_FILE" ]; then
  echo "Missing $PARSER_FILE" >&2
  exit 1
fi

# NVIDIA's vLLM example for this checkpoint uses the qwen3_coder tool parser.
mkdir -p "$HF_HOME_LOCAL" "$VLLM_CACHE_LOCAL"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
  --gpus all \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  -p "$PORT:$PORT" \
  --ipc=host \
  -v "$HF_HOME_LOCAL:/root/.cache/huggingface" \
  -v "$VLLM_CACHE_LOCAL:/root/.cache/vllm" \
  -v "$PARSER_FILE:/opt/spark-setup/super_v3_reasoning_parser.py:ro" \
  -e HF_HUB_OFFLINE="$NEMOTRON_HF_HUB_OFFLINE" \
  -e TRANSFORMERS_OFFLINE="$NEMOTRON_TRANSFORMERS_OFFLINE" \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  scitrera/dgx-spark-vllm:0.16.0-t5 \
  vllm serve "$MODEL_NAME" \
    --port "$PORT" \
    --async-scheduling \
    --dtype auto \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --swap-space 0 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization "$NEMOTRON_GPU_MEMORY_UTILIZATION" \
    --attention-backend "$NEMOTRON_ATTENTION_BACKEND" \
    --block-size "$NEMOTRON_BLOCK_SIZE" \
    --enable-chunked-prefill \
    --max-num-batched-tokens "$NEMOTRON_MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$NEMOTRON_MAX_NUM_SEQS" \
    --max-model-len "$NEMOTRON_MAX_MODEL_LEN" \
    --trust-remote-code \
    --disable-log-requests \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser-plugin /opt/spark-setup/super_v3_reasoning_parser.py \
    --reasoning-parser super_v3

echo "Container $CONTAINER_NAME started on port $PORT."
echo "HF_HUB_OFFLINE=$NEMOTRON_HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$NEMOTRON_TRANSFORMERS_OFFLINE"
echo "Use 'docker logs -f $CONTAINER_NAME' to follow logs."
