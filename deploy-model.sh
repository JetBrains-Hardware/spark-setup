#!/bin/bash
set -euo pipefail

# Simple remote deploy for one text model on one DGX Spark.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE="${REMOTE:-}"
REMOTE_DIR="${REMOTE_DIR:-spark-setup}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)

MODEL_KIND=""
COPY_ONLY=false
START_ONLY=false
DRY_RUN=false

usage() {
  cat <<EOF
Usage: REMOTE=user@host $0 <qwen|gpt-oss|nemotron3> [--copy-only] [--start-only] [--dry-run]
EOF
}

for arg in "$@"; do
  case "$arg" in
    qwen|gpt-oss|nemotron3)
      if [ -n "$MODEL_KIND" ]; then
        echo "Model already set to $MODEL_KIND" >&2
        exit 2
      fi
      MODEL_KIND="$arg"
      ;;
    --copy-only)
      COPY_ONLY=true
      ;;
    --start-only)
      START_ONLY=true
      ;;
    --dry-run)
      DRY_RUN=true
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

if [ -z "$MODEL_KIND" ]; then
  usage >&2
  exit 2
fi

if [ -z "$REMOTE" ]; then
  echo "Set REMOTE=user@host" >&2
  usage >&2
  exit 2
fi

if [ "$COPY_ONLY" = true ] && [ "$START_ONLY" = true ]; then
  echo "Use only one of --copy-only or --start-only" >&2
  exit 2
fi

MODEL_NAME=""
START_CMD=""
WAIT_CMD=""
SMOKE_CMD=""
REMOTE_DIRS=""
MODEL_FILES=()
STOP_PEER_CONTAINERS_CMD="docker rm -f \
  vllm_qwen_code gpt-oss vllm_nemotron3 \
  vllm_qwen_vl vllm_lfm flux_image comfyui_spark \
  >/dev/null 2>&1 || true"
COMMON_REMOTE_DIRS="~/$REMOTE_DIR ~/.cache/huggingface ~/.cache/vllm"

case "$MODEL_KIND" in
  qwen)
    MODEL_NAME="Qwen3-Coder-Next"
    START_CMD="HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 bash run-qwen3.sh"
    WAIT_CMD="curl -sf --max-time 5 http://localhost:8000/health"
    SMOKE_CMD="cd ~/$REMOTE_DIR && bash qwen3-load.sh localhost:8000 >/dev/null"
    REMOTE_DIRS="$COMMON_REMOTE_DIRS"
    MODEL_FILES=(
      run-qwen3.sh
      qwen3-load.sh
    )
    ;;
  gpt-oss)
    MODEL_NAME="GPT-OSS"
    START_CMD="PORT=8001 CONTAINER_NAME=gpt-oss bash run-gpt-oss.sh"
    WAIT_CMD="curl -sf --max-time 5 http://localhost:8001/v1/responses \
      -H 'Content-Type: application/json' \
      -d '{\"model\":\"openai/gpt-oss-120b\",\"input\":\"ping\"}'"
    SMOKE_CMD="cd ~/$REMOTE_DIR && bash gpt-oss-load.sh localhost:8001 >/dev/null"
    REMOTE_DIRS="$COMMON_REMOTE_DIRS ~/.cache/triton"
    MODEL_FILES=(
      run-gpt-oss.sh
      gpt-oss-load.sh
      Dockerfile.gpt-oss
      in-container.sh
    )
    ;;
  nemotron3)
    MODEL_NAME="Nemotron 3"
    START_CMD="PORT=8003 CONTAINER_NAME=vllm_nemotron3 bash run-nemotron3.sh"
    WAIT_CMD="curl -sf --max-time 5 http://localhost:8003/health"
    SMOKE_CMD="cd ~/$REMOTE_DIR && bash nemotron3-load.sh localhost:8003 >/dev/null"
    REMOTE_DIRS="$COMMON_REMOTE_DIRS"
    MODEL_FILES=(
      run-nemotron3.sh
      nemotron3-load.sh
    )
    ;;
esac

run_cmd() {
  if [ "$DRY_RUN" = true ]; then
    printf '+ %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

warn_old_services() {
  if [ "$DRY_RUN" = true ]; then
    return 0
  fi

  local services
  services="$(ssh "${SSH_OPTS[@]}" "$REMOTE" \
    'systemctl list-units --type=service --state=running 2>/dev/null | grep -E "qwen-server|qwen3-vllm|qwen3-coder-server|gpt-oss-server" || true')"
  if [ -n "$services" ]; then
    echo "Warning: old systemd services are still running on $REMOTE" >&2
    echo "$services" >&2
    echo "Disable them with sudo on the Spark if they interfere with Docker restarts." >&2
  fi
}

copy_files() {
  local tmpdir=""

  tmpdir="$(mktemp -d)"
  cleanup_tmpdir() {
    if [ -n "${tmpdir:-}" ] && [ -d "$tmpdir" ]; then
      rm -rf "$tmpdir"
    fi
  }
  trap cleanup_tmpdir RETURN

  local file
  for file in "${MODEL_FILES[@]}"; do
    sed 's/\r$//' "$SCRIPT_DIR/$file" > "$tmpdir/$file"
  done

  run_cmd ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p $REMOTE_DIRS"

  local scp_files=()
  for file in "${MODEL_FILES[@]}"; do
    scp_files+=("$tmpdir/$file")
  done
  run_cmd scp "${SSH_OPTS[@]}" "${scp_files[@]}" "$REMOTE:~/$REMOTE_DIR/"

  run_cmd ssh "${SSH_OPTS[@]}" "$REMOTE" "chmod +x ~/$REMOTE_DIR/*.sh"
}

wait_for_remote_cmd() {
  local name="$1"
  local cmd="$2"
  local attempts="${3:-180}"

  if [ "$DRY_RUN" = true ]; then
    printf '+ wait for %s using remote command: %s\n' "$name" "$cmd"
    return 0
  fi

  echo "Waiting for $name ..."
  run_cmd ssh "${SSH_OPTS[@]}" "$REMOTE" \
    "for i in \$(seq 1 $attempts); do $cmd >/dev/null 2>&1 && exit 0; sleep 5; done; exit 1"
}

start_model() {
  local hf_export=""
  if [ -n "${HF_TOKEN:-}" ]; then
    hf_export="export HF_TOKEN=$(printf '%q' "$HF_TOKEN") && "
  fi

  warn_old_services
  run_cmd ssh "${SSH_OPTS[@]}" "$REMOTE" "$STOP_PEER_CONTAINERS_CMD"
  run_cmd ssh "${SSH_OPTS[@]}" "$REMOTE" "${hf_export}cd ~/$REMOTE_DIR && $START_CMD"

  wait_for_remote_cmd "$MODEL_NAME health" "$WAIT_CMD"

  echo "Running smoke test for $MODEL_NAME ..."
  run_cmd ssh "${SSH_OPTS[@]}" "$REMOTE" "$SMOKE_CMD"
}

if [ "$START_ONLY" = false ]; then
  echo "Copying $MODEL_NAME files to $REMOTE:~/$REMOTE_DIR ..."
  copy_files
fi

if [ "$COPY_ONLY" = false ]; then
  echo "Starting $MODEL_NAME on $REMOTE ..."
  start_model
fi

echo "$MODEL_NAME deploy finished."
