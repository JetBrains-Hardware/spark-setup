#!/bin/bash

# Quick smoke test for GPT-OSS Responses API.
# Usage: ./gpt-oss-load.sh [host:port]

set -euo pipefail

ENDPOINT="${1:-localhost:8001}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
RESPONSE_FILE="$(mktemp)"
trap 'rm -f "$RESPONSE_FILE"' EXIT

echo "Testing GPT-OSS at $ENDPOINT ..."

curl -sf --max-time "$REQUEST_TIMEOUT" "http://$ENDPOINT/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "input": "Write one short sentence saying the endpoint works."
  }' >"$RESPONSE_FILE"

python3 -m json.tool <"$RESPONSE_FILE" 2>/dev/null || cat "$RESPONSE_FILE"

echo ""
