#!/bin/bash

set -e -x

# Model configuration - pin the version
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-120b}"
MODEL_REVISION="${MODEL_REVISION:-b5c939de8f754692c1647ca79fbf85e8c1e70f8a}"
PORT="${PORT:-8001}"
MODEL_PATH_NAME="${MODEL_NAME//\//--}"

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

# Function to send OS notification (fails silently if no X server)
send_notification() {
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

# Download weights if missing; allow online mode only for the download
if [[ ! -d "$MODEL_SNAPSHOT" ]] || ! ls "$MODEL_SNAPSHOT"/model-*.safetensors >/dev/null 2>&1; then
    mkdir -p "$MODEL_SNAPSHOT"
    HF_HUB_OFFLINE=0 huggingface-cli download "$MODEL_NAME" --revision "$MODEL_REVISION" --local-dir "$MODEL_SNAPSHOT" --local-dir-use-symlinks False
fi

if ! ls "$MODEL_SNAPSHOT"/model-*.safetensors >/dev/null 2>&1; then
    echo "Model files are missing under $MODEL_SNAPSHOT" >&2
    exit 1
fi

# Force offline afterwards to avoid hub calls during load
export HF_HUB_OFFLINE=1

sed -i 's~REASONING_EFFORT = ReasoningEffort.LOW~REASONING_EFFORT = "medium"~' /usr/local/lib/python3.12/dist-packages/gpt_oss/responses_api/types.py
sed -i 's~port=args.port~port=args.port, host="0.0.0.0"~' /usr/local/lib/python3.12/dist-packages/gpt_oss/responses_api/serve.py
# Increase GPU memory utilization to 0.85 while leaving some system headroom.
# Use runai_streamer for faster model loading (parallel weight loading)
sed -i 's/llm = LLM(/llm = LLM(\n        gpu_memory_utilization=0.85,\n        load_format="runai_streamer",/' /usr/local/lib/python3.12/dist-packages/gpt_oss/responses_api/inference/vllm.py
sed -i 's~tensor_parallel_size=TP~tensor_parallel_size=1~' /usr/local/lib/python3.12/dist-packages/gpt_oss/responses_api/inference/vllm.py

ensure_line_present() {
    local line="$1"
    local file="$2"

    grep -Fqx "$line" "$file" || echo "$line" >> "$file"
}

GPT_TYPES_FILE="/usr/local/lib/python3.12/dist-packages/gpt_oss/responses_api/types.py"
ensure_line_present '    temperature: float = 0' "$GPT_TYPES_FILE"
ensure_line_present '    tools: list = []' "$GPT_TYPES_FILE"
ensure_line_present '    reasoning: None = None' "$GPT_TYPES_FILE"
ensure_line_present '    truncation: None = None' "$GPT_TYPES_FILE"
ensure_line_present '    user: None = None' "$GPT_TYPES_FILE"

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
