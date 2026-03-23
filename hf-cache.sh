#!/bin/bash

hf_have_cached_weights() {
  local model_name="$1"
  local hf_home="${2:-$HOME/.cache/huggingface}"
  local model_path_name="${model_name//\//--}"
  local model_cache_dir="$hf_home/hub/models--${model_path_name}"
  local snapshot_dir=""
  local shard_count=0

  for snapshot_dir in "$model_cache_dir"/snapshots/*; do
    [ -d "$snapshot_dir" ] || continue

    shard_count="$(find "$snapshot_dir" -maxdepth 1 -name '*.safetensors' | wc -l | tr -d ' ')"
    [ "${shard_count:-0}" -gt 0 ] || continue
    [ -f "$snapshot_dir/config.json" ] || continue
    if [ ! -f "$snapshot_dir/tokenizer.json" ] && \
       [ ! -f "$snapshot_dir/tokenizer_config.json" ] && \
       [ ! -f "$snapshot_dir/tokenizer.model" ]; then
      continue
    fi

    if [ -f "$snapshot_dir/model.safetensors.index.json" ]; then
      local expected_shards
      expected_shards="$(python3 -c "
import json, sys
with open('$snapshot_dir/model.safetensors.index.json') as f:
    print(len(set(json.load(f)['weight_map'].values())))
" 2>/dev/null || echo 0)"
      if [ "${expected_shards:-0}" -gt 0 ] && [ "$shard_count" -lt "$expected_shards" ]; then
        continue
      fi
    fi

    return 0
  done

  return 1
}

hf_pick_offline_mode() {
  local model_name="$1"
  local hf_home="${2:-$HOME/.cache/huggingface}"
  local explicit="${3:-}"

  if [ -n "$explicit" ]; then
    printf '%s\n' "$explicit"
  elif hf_have_cached_weights "$model_name" "$hf_home"; then
    printf '1\n'
  else
    printf '0\n'
  fi
}
