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
FORCE_REBUILD="${FORCE_REBUILD:-0}"
ENABLE_NOTIFICATIONS="${ENABLE_NOTIFICATIONS:-0}"
DEBUG_STARTUP="${DEBUG_STARTUP:-0}"

mkdir -p "$HF_HOME_LOCAL" "$VLLM_CACHE_LOCAL" "$TRITON_CACHE_LOCAL"

cd "$SCRIPT_DIR"
if [ "$FORCE_REBUILD" = "1" ] || ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    docker build -t "$IMAGE_NAME" -f Dockerfile.gpt-oss .
fi
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker_args=(
    -d
    --network host
    --restart unless-stopped
    --shm-size 10.24g
    --gpus all
    -v "$HF_HOME_LOCAL:/root/.cache/huggingface"
    -v "$HF_HOME_LOCAL:/hub"
    -v "$VLLM_CACHE_LOCAL:/root/.cache/vllm"
    -v "$TRITON_CACHE_LOCAL:/root/.cache/triton"
    -e PORT="$PORT"
    -e MODEL_NAME="$MODEL_NAME"
    -e MODEL_REVISION="$MODEL_REVISION"
    -e ENABLE_NOTIFICATIONS="$ENABLE_NOTIFICATIONS"
    -e DEBUG_STARTUP="$DEBUG_STARTUP"
    --name "$CONTAINER_NAME"
)

if [ "$ENABLE_NOTIFICATIONS" = "1" ]; then
    docker_args+=(
        -v /tmp/.X11-unix:/tmp/.X11-unix
        -e DISPLAY="${DISPLAY:-:0}"
    )
fi

docker run "${docker_args[@]}" \
    ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
    "$IMAGE_NAME"

echo "Container $CONTAINER_NAME started on port $PORT."
echo "Use 'docker logs -f $CONTAINER_NAME' to follow logs."
