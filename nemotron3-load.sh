#!/bin/bash
set -euo pipefail

ENDPOINT="${1:-localhost:8003}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
RESPONSE_FILE="$(mktemp)"
trap 'rm -f "$RESPONSE_FILE"' EXIT

curl -sf --max-time "$REQUEST_TIMEOUT" "http://$ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "messages": [{"role": "user", "content": "Say the endpoint works."}],
    "chat_template_kwargs": {"enable_thinking": false, "force_nonempty_content": true},
    "max_tokens": 64
  }' >"$RESPONSE_FILE"

python3 -m json.tool <"$RESPONSE_FILE" 2>/dev/null || cat "$RESPONSE_FILE"
echo ""
