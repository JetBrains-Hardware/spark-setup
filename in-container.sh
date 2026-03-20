#!/bin/bash

set -e

if [ "${DEBUG_STARTUP:-0}" = "1" ]; then
    set -x
fi

# Model configuration - pin the version
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-120b}"
MODEL_REVISION="${MODEL_REVISION:-b5c939de8f754692c1647ca79fbf85e8c1e70f8a}"
PORT="${PORT:-8001}"
MODEL_PATH_NAME="${MODEL_NAME//\//--}"
ENABLE_NOTIFICATIONS="${ENABLE_NOTIFICATIONS:-0}"

# Force HF cache into the mounted path (see run.sh volumes)
export HF_HOME="/hub"
mkdir -p "${HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export VLLM_DOWNLOAD_DIR="${VLLM_DOWNLOAD_DIR:-$HF_HOME}"
export VLLM_WORKER_MULTIPROCESS_LOAD="${VLLM_WORKER_MULTIPROCESS_LOAD:-1}"
# Performance optimization: Enable CUDA graphs for better GPU utilization
# export VLLM_ENFORCE_EAGER=1  # Disabled to allow CUDA graph compilation
# export TORCHDYNAMO_DISABLE=1  # Disabled to allow torch.compile optimizations
MODEL_SNAPSHOT="${HF_HOME}/hub/models--${MODEL_PATH_NAME}/snapshots/${MODEL_REVISION}"
MODEL_INCLUDE_PATTERNS=(
    "chat_template.jinja"
    "config.json"
    "generation_config.json"
    "model-*.safetensors"
    "model.safetensors.index.json"
    "tokenizer.json"
    "tokenizer_config.json"
)

snapshot_has_complete_weights() {
    local snapshot="$1"
    local actual_shards=0
    local expected_shards=0

    [ -d "$snapshot" ] || return 1
    [ -f "$snapshot/config.json" ] || return 1
    if [ ! -f "$snapshot/tokenizer.json" ] && [ ! -f "$snapshot/tokenizer_config.json" ] && \
       [ ! -f "$snapshot/tokenizer.model" ]; then
        return 1
    fi

    actual_shards="$(find "$snapshot" -maxdepth 1 -name 'model-*.safetensors' | wc -l | tr -d ' ')"
    [ "${actual_shards:-0}" -gt 0 ] || return 1

    if [ -f "$snapshot/model.safetensors.index.json" ]; then
        expected_shards="$(python3 - "$snapshot/model.safetensors.index.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    data = json.load(fh)

print(len(set(data.get("weight_map", {}).values())))
PY
)"
        [ "${expected_shards:-0}" -gt 0 ] || return 1
        [ "$actual_shards" -ge "$expected_shards" ] || return 1
    fi

    return 0
}

# Function to send OS notification (fails silently if no X server)
send_notification() {
    if [ "$ENABLE_NOTIFICATIONS" != "1" ]; then
        return 0
    fi

    local title="$1"
    local message="$2"
    local urgency="${3:-normal}"  # low, normal, critical

    # Try to use DISPLAY :0 if not set (common for X server)
    export DISPLAY="${DISPLAY:-:0}"

    # Try notify-send if available, suppress all errors
    if command -v notify-send &>/dev/null; then
        notify-send --urgency="$urgency" "$title" "$message" 2>/dev/null || true
    fi
}

# Function to wait for the target port to be ready and send notification
wait_for_port_ready() {
    local port="$PORT"
    local max_attempts=300  # 5 minutes max wait
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if command -v curl &>/dev/null; then
            if curl -s --max-time 1 "http://localhost:$port" &>/dev/null; then
                send_notification "GPT Server Ready" "Port $port is now responding. Server is ready to accept requests." "normal"
                return 0
            fi
        elif command -v nc &>/dev/null; then
            if nc -z localhost $port 2>/dev/null; then
                send_notification "GPT Server Ready" "Port $port is now responding. Server is ready to accept requests." "normal"
                return 0
            fi
        fi

        sleep 1
        ((attempt++))
    done

    return 1
}

# === One-time setup ===

# Download weights if missing or incomplete; allow online mode only for the download
if ! snapshot_has_complete_weights "$MODEL_SNAPSHOT"; then
    mkdir -p "$MODEL_SNAPSHOT"
    HF_HUB_OFFLINE=0 python3 - "$MODEL_NAME" "$MODEL_REVISION" "$MODEL_SNAPSHOT" "${MODEL_INCLUDE_PATTERNS[@]}" <<'PY'
import sys

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
revision = sys.argv[2]
local_dir = sys.argv[3]
allow_patterns = sys.argv[4:]

snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=local_dir,
    allow_patterns=allow_patterns,
)
PY
fi

if ! snapshot_has_complete_weights "$MODEL_SNAPSHOT"; then
    echo "Model snapshot is incomplete under $MODEL_SNAPSHOT" >&2
    exit 1
fi

# Force offline afterwards to avoid hub calls during load
export HF_HUB_OFFLINE=1

# === Server loop ===

echo "Starting GPT-OSS server in loop mode..."
echo "Model: $MODEL_NAME @ $MODEL_REVISION"
echo "Checkpoint: $MODEL_SNAPSHOT"
echo "Port: $PORT"

while true; do
    echo "$(date): Starting python server..."
    send_notification "GPT Server Starting" "Python server is starting up. Loading model..." "low"

    # Start background process to detect when port is ready
    wait_for_port_ready &
    PORT_CHECKER_PID=$!

    set +e
    python3 -m gpt_oss.responses_api.serve \
        --checkpoint "$MODEL_SNAPSHOT" \
        --port "$PORT" \
        --inference-backend vllm
    EXIT_CODE=$?
    set -e
    echo "$(date): Python server exited with code $EXIT_CODE"

    # Kill port checker if still running
    kill $PORT_CHECKER_PID 2>/dev/null || true

    send_notification "GPT Server Stopped" "Python server exited with code $EXIT_CODE. Restarting in 1 second..." "critical"

    echo "Sleeping for 1 second before restart..."
    sleep 1
done
