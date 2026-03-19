#!/bin/bash

set -euo pipefail

# Run GPT-OSS in detached mode on the Spark.
#
# Defaults:
#   - image: gpt-oss-custom:latest
#   - container: gpt-oss
#   - port: 8001 (so Qwen can stay on 8000)
#   - model: openai/gpt-oss-120b
#
# The container entrypoint loops the server process, so detached mode is enough.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-gpt-oss-custom:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-gpt-oss}"
PORT="${PORT:-8001}"
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-120b}"
MODEL_REVISION="${MODEL_REVISION:-b5c939de8f754692c1647ca79fbf85e8c1e70f8a}"
HF_HOME_LOCAL="${HF_HOME_LOCAL:-$HOME/.cache/huggingface}"
VLLM_CACHE_LOCAL="${VLLM_CACHE_LOCAL:-$HOME/.cache/vllm}"
TRITON_CACHE_LOCAL="${TRITON_CACHE_LOCAL:-$HOME/.cache/triton}"

mkdir -p "$HF_HOME_LOCAL" "$VLLM_CACHE_LOCAL" "$TRITON_CACHE_LOCAL"

cd "$SCRIPT_DIR"
docker build -t "$IMAGE_NAME" -f Dockerfile.gpt-oss .
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
    --network host \
    --restart unless-stopped \
    --shm-size 10.24g \
    --gpus all \
    -v "$HF_HOME_LOCAL:/root/.cache/huggingface" \
    -v "$HF_HOME_LOCAL:/hub" \
    -v "$VLLM_CACHE_LOCAL:/root/.cache/vllm" \
    -v "$TRITON_CACHE_LOCAL:/root/.cache/triton" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e PORT="$PORT" \
    -e MODEL_NAME="$MODEL_NAME" \
    -e MODEL_REVISION="$MODEL_REVISION" \
    ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
    --name "$CONTAINER_NAME" \
    "$IMAGE_NAME"

echo "Container $CONTAINER_NAME started on port $PORT."
echo "Use 'docker logs -f $CONTAINER_NAME' to follow logs."
