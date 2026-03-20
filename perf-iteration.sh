#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-}"
ITERATION_NAME="${2:-}"

if [ -z "$MODEL" ] || [ -z "$ITERATION_NAME" ]; then
  echo "Usage: REMOTE=user@host $0 <qwen|gpt-oss|nemotron3> <iteration-name> [deploy flags...]" >&2
  exit 2
fi

shift 2

if [ -z "${REMOTE:-}" ]; then
  echo "Set REMOTE=user@host" >&2
  exit 2
fi

case "$MODEL" in
  qwen)
    WRAPPER="$SCRIPT_DIR/deploy-qwen3.sh"
    CAPTURE_VARS=(
      QWEN_ATTENTION_BACKEND
      QWEN_GPU_MEMORY_UTILIZATION
      QWEN_MAX_NUM_SEQS
      QWEN_MAX_NUM_BATCHED_TOKENS
      QWEN_BLOCK_SIZE
      QWEN_MAX_MODEL_LEN
      HF_HUB_OFFLINE
      TRANSFORMERS_OFFLINE
    )
    ;;
  gpt-oss)
    WRAPPER="$SCRIPT_DIR/deploy-gpt-oss.sh"
    CAPTURE_VARS=(
      FORCE_REBUILD
      ENABLE_NOTIFICATIONS
      DEBUG_STARTUP
    )
    ;;
  nemotron3)
    WRAPPER="$SCRIPT_DIR/deploy-nemotron3.sh"
    CAPTURE_VARS=(
      NEMOTRON_GPU_MEMORY_UTILIZATION
      NEMOTRON_ATTENTION_BACKEND
      NEMOTRON_BLOCK_SIZE
      NEMOTRON_MAX_NUM_BATCHED_TOKENS
      NEMOTRON_MAX_NUM_SEQS
      NEMOTRON_MAX_MODEL_LEN
      HF_HUB_OFFLINE
      TRANSFORMERS_OFFLINE
    )
    ;;
  *)
    echo "Unknown model: $MODEL" >&2
    exit 2
    ;;
esac

RUN_ROOT="${PERF_RUNS_DIR:-$SCRIPT_DIR/perf-runs}"
RUN_ID="$(date -u +%Y%m%d-%H%M%S)-${MODEL}-${ITERATION_NAME}"
RUN_DIR="$RUN_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

{
  echo "MODEL=$MODEL"
  echo "ITERATION=$ITERATION_NAME"
  echo "REMOTE=$REMOTE"
  echo "STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "GIT_HEAD=$(git -C "$SCRIPT_DIR" rev-parse HEAD)"
  echo "HF_TOKEN_SET=$([ -n "${HF_TOKEN:-}" ] && echo yes || echo no)"
} > "$RUN_DIR/meta.txt"

git -C "$SCRIPT_DIR" status --short > "$RUN_DIR/git-status.txt"
for env_name in "${CAPTURE_VARS[@]}"; do
  if [ "${!env_name+x}" = x ]; then
    printf '%s=%s\n' "$env_name" "${!env_name}" >> "$RUN_DIR/env.txt"
  fi
done
printf '%q ' "$WRAPPER" "$@" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"

set +e
"$WRAPPER" "$@" 2>&1 | tee "$RUN_DIR/deploy.log"
exit_code=${PIPESTATUS[0]}
set -e

echo "EXIT_CODE=$exit_code" >> "$RUN_DIR/meta.txt"
echo "FINISHED=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_DIR/meta.txt"
echo "Artifacts: $RUN_DIR"
exit "$exit_code"
