#!/bin/bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-nvidia/nemotron-3-8b-chat-4k-sft}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_nemotron3}"
PORT="${PORT:-8003}"

mkdir -p ~/.cache/huggingface ~/.cache/vllm
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
  --gpus all \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  -p "$PORT:$PORT" \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  scitrera/dgx-spark-vllm:0.16.0-t5 \
  vllm serve "$MODEL_NAME" \
    --port "$PORT" \
    --dtype auto \
    --gpu-memory-utilization 0.20 \
    --max-model-len 4096 \
    --trust-remote-code \
    --disable-log-requests
