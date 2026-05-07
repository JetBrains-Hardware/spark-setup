#!/bin/bash
set -euo pipefail

# Smoke test for vLLM Qwen3.6-27B-FP8 endpoint.
# Usage: ./qwen36-load.sh [host:port]

ENDPOINT="${1:-localhost:8005}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
RESPONSE_FILE="$(mktemp)"
trap 'rm -f "$RESPONSE_FILE"' EXIT

echo "Testing vLLM at $ENDPOINT ..."

echo "=== Health ==="
curl -sf --max-time "$REQUEST_TIMEOUT" "http://$ENDPOINT/health"
echo ""

echo "=== Chat Completion ==="
curl -sf --max-time "$REQUEST_TIMEOUT" "http://$ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.6-27B-FP8",
    "messages": [{"role": "user", "content": "Say the endpoint works."}],
    "max_tokens": 64
  }' >"$RESPONSE_FILE"

python3 -m json.tool <"$RESPONSE_FILE" 2>/dev/null || cat "$RESPONSE_FILE"
echo ""
