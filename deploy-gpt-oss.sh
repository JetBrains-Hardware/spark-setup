#!/bin/bash
set -euo pipefail

exec "$(dirname "$0")/deploy-model.sh" gpt-oss "$@"
