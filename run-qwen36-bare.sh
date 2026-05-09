#!/bin/bash
set -euo pipefail

# Bare-metal Qwen3.6-27B-FP8 launcher — runs vLLM directly from a venv on the Spark
# (no Docker). Companion to run-qwen36.sh; serves on :8006 by default so it can run
# alongside the containerized version on :8005 for A/B comparison.
#
# Why bare metal: lets us pin a newer vLLM (0.20+) without rebuilding the community
# image, lets us pip-install community forks (DFlash, monkey-patches), and removes
# the FS bind-mount permission gotchas around ~/.cache/huggingface.
#
# Tracks the official vLLM recipe at https://recipes.vllm.ai/Qwen/Qwen3.6-27B and
# adds Spark-tuned defaults that survived our Phase 1–3 perf passes.
#
# Useful env knobs (all optional):
#   QWEN36_BARE_VENV                   path to venv (default ~/spark-setup-baremetal/.venv)
#   QWEN36_BARE_PORT                   server port (default 8006)
#   QWEN36_BARE_MAX_MODEL_LEN          context window cap (default 65536; bump to 262144 for full 256k)
#   QWEN36_BARE_GPU_MEMORY_UTILIZATION default 0.80
#   QWEN36_BARE_MAX_NUM_SEQS           default 32
#   QWEN36_BARE_MAX_NUM_BATCHED_TOKENS default 8192
#   QWEN36_BARE_BLOCK_SIZE             default 16
#   QWEN36_BARE_NUM_SPECULATIVE_TOKENS default 1 (Phase 2 winner; 0 to disable; >=2 risks engine crash on this hardware)
#   QWEN36_BARE_KV_CACHE_DTYPE         default auto; set to fp8 to halve KV memory (needed for 200k+ contexts)
#   QWEN36_BARE_DRAFT_METHOD           speculative method override (default mtp; e.g. dflash)
#   QWEN36_BARE_DRAFT_MODEL            draft model HF id when method != mtp (e.g. z-lab/Qwen3.6-27B-DFlash)
#   QWEN36_BARE_LOG                    log file (default ~/spark-setup-baremetal/vllm.log)

VENV="${QWEN36_BARE_VENV:-$HOME/spark-setup-baremetal/.venv}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B-FP8}"
PORT="${QWEN36_BARE_PORT:-8006}"
MAX_MODEL_LEN="${QWEN36_BARE_MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${QWEN36_BARE_GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${QWEN36_BARE_MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${QWEN36_BARE_MAX_NUM_BATCHED_TOKENS:-16384}"
BLOCK_SIZE="${QWEN36_BARE_BLOCK_SIZE:-32}"
NUM_SPECULATIVE_TOKENS="${QWEN36_BARE_NUM_SPECULATIVE_TOKENS:-1}"
KV_CACHE_DTYPE="${QWEN36_BARE_KV_CACHE_DTYPE:-fp8}"
DRAFT_METHOD="${QWEN36_BARE_DRAFT_METHOD:-mtp}"
DRAFT_MODEL="${QWEN36_BARE_DRAFT_MODEL:-}"
LOG="${QWEN36_BARE_LOG:-$HOME/spark-setup-baremetal/vllm.log}"
ATTENTION_BACKEND="${QWEN36_BARE_ATTENTION_BACKEND:-FLASH_ATTN}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "ERROR: venv missing at $VENV — create it first (python3 -m venv $VENV && pip install vllm)" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG")"

# Stop any previous bare-metal vllm process listening on the same port.
if pids="$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null)" && [ -n "$pids" ]; then
  echo "stopping prior listener on :$PORT (pids: $pids)"
  kill $pids 2>/dev/null || true
  sleep 2
  kill -9 $pids 2>/dev/null || true
fi

extra_args=()
if [ "$NUM_SPECULATIVE_TOKENS" != 0 ]; then
  if [ "$DRAFT_METHOD" = mtp ]; then
    extra_args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS}}")
  else
    if [ -z "$DRAFT_MODEL" ]; then
      echo "ERROR: DRAFT_METHOD=$DRAFT_METHOD requires QWEN36_BARE_DRAFT_MODEL" >&2
      exit 2
    fi
    extra_args+=(--speculative-config "{\"method\":\"${DRAFT_METHOD}\",\"model\":\"${DRAFT_MODEL}\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS}}")
  fi
fi
if [ "$KV_CACHE_DTYPE" != auto ]; then
  extra_args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

# vLLM env knobs validated on GB10 in the Docker run.
export VLLM_USE_DEEP_GEMM=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND="$ATTENTION_BACKEND"

cd "$(dirname "$VENV")"
nohup "$VENV/bin/vllm" serve "$MODEL_NAME" \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --block-size "$BLOCK_SIZE" \
  --enable-prefix-caching \
  --max-model-len "$MAX_MODEL_LEN" \
  --reasoning-parser qwen3 \
  "${extra_args[@]}" \
  >>"$LOG" 2>&1 &

PID=$!
echo "$PID" > "$VENV/../vllm.pid"
echo "vllm started (pid $PID), logs: $LOG"
echo "speculative: method=$DRAFT_METHOD num=$NUM_SPECULATIVE_TOKENS draft=${DRAFT_MODEL:-<built-in>}"
echo "kv-cache-dtype=$KV_CACHE_DTYPE max-model-len=$MAX_MODEL_LEN"
echo "tail logs: tail -f $LOG"
