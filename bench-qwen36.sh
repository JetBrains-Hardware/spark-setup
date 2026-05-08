#!/bin/bash
set -euo pipefail

# bench-qwen36.sh — run llama-benchy against a running Qwen3.6-27B-FP8 endpoint
# and emit a results bundle under perf-runs/.
#
# Run this on the Spark itself; running from a remote host adds LAN latency
# to TTFR/TTFT and distorts comparisons across runs.
#
# Profiles:
#   decode     — single-stream tg @ depth=0 (interactive responsiveness)
#   throughput — aggregate t/s @ concurrency 1/4/8 (server load)
#   longctx    — TTFT @ depth 0 / 8k / 32k (repo-level prompts)
#   huge       — TTFT @ depth 0 / 64k / 131k / 200k (200k+ context, needs KV-compact run)
#   all        — run decode + throughput + longctx (skips huge — opt-in)
#
# Usage:
#   bash bench-qwen36.sh <decode|throughput|longctx|huge|all> [iteration-name]

PROFILE="${1:-}"
ITER="${2:-$(date -u +%Y%m%d-%H%M%S)}"

if [ -z "$PROFILE" ]; then
  echo "Usage: $0 <decode|throughput|longctx|huge|all> [iteration-name]" >&2
  exit 2
fi

ENDPOINT="${ENDPOINT:-http://localhost:8005/v1}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
RUNS="${RUNS:-3}"
OUT_ROOT="${PERF_RUNS_DIR:-$HOME/spark-setup/perf-runs}"
OUT_DIR="$OUT_ROOT/qwen36-bench-$ITER"
mkdir -p "$OUT_DIR"

PATH="$HOME/.local/bin:$PATH"
if ! command -v llama-benchy >/dev/null 2>&1; then
  echo "llama-benchy not on PATH (or in ~/.local/bin). Install with:" >&2
  echo "  uv pip install -U --system llama-benchy" >&2
  echo "  # or: python3 -m pip install --user --break-system-packages -U llama-benchy" >&2
  exit 1
fi

# Force offline + reuse the model's own tokenizer instead of fetching gpt2 (the default).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
TOKENIZER="${TOKENIZER:-$MODEL}"

run_one() {
  local name="$1"; shift
  local log="$OUT_DIR/$name.log"
  echo "===== $name =====" | tee -a "$log"
  echo "+ llama-benchy $*" | tee -a "$log"
  llama-benchy "$@" 2>&1 | tee -a "$log"
}

decode_profile() {
  run_one decode \
    --base-url "$ENDPOINT" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --pp 512 \
    --tg 256 \
    --depth 0 \
    --runs "$RUNS" \
    --concurrency 1 \
    --latency-mode generation
}

throughput_profile() {
  run_one throughput \
    --base-url "$ENDPOINT" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --pp 1024 \
    --tg 256 \
    --depth 0 \
    --runs "$RUNS" \
    --concurrency 1 4 8 \
    --latency-mode generation
}

longctx_profile() {
  run_one longctx \
    --base-url "$ENDPOINT" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --pp 512 \
    --tg 32 \
    --depth 0 8192 32768 \
    --runs "$RUNS" \
    --concurrency 1 \
    --latency-mode generation
}

huge_profile() {
  run_one huge \
    --base-url "$ENDPOINT" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --pp 256 \
    --tg 16 \
    --depth 0 65536 131072 200000 \
    --runs 2 \
    --concurrency 1 \
    --latency-mode generation
}

case "$PROFILE" in
  decode)     decode_profile ;;
  throughput) throughput_profile ;;
  longctx)    longctx_profile ;;
  huge)       huge_profile ;;
  all)        decode_profile; throughput_profile; longctx_profile ;;
  *) echo "Unknown profile: $PROFILE" >&2; exit 2 ;;
esac

echo "Results: $OUT_DIR"
