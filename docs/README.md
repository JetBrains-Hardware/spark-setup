# Deployment Notes

## Remote Contract

The deploy scripts assume:

- `REMOTE=user@host` points at the Spark
- the remote project directory is `~/spark-setup` unless `REMOTE_DIR` overrides it
- Docker commands run without `sudo`
- model caches live under `~/.cache/huggingface`, `~/.cache/vllm`, and `~/.cache/triton`

## Model Differences

### Qwen

- starts on `:8000`
- serves OpenAI chat completions
- starts offline automatically when a cached snapshot already exists
- tuned for the larger coder workload and longer context

### Qwen3.6

- starts on `:8005`
- serves OpenAI chat completions, reasoning content via `--reasoning-parser qwen3`
- uses the dense FP8 checkpoint `Qwen/Qwen3.6-27B-FP8`
- follows the official vLLM recipe (`--max-model-len 262144 --reasoning-parser qwen3 --enable-prefix-caching`)
- ships native MTP (multi-token-prediction) heads — opt in via `QWEN36_NUM_SPECULATIVE_TOKENS=1`; no separate draft model needed
- offline-cache-aware via the same `hf-cache.sh` helper as Qwen and Nemotron

### GPT-OSS

- starts on `:8001`
- serves the OpenAI Responses API
- builds a custom container image before startup
- patches the installed `gpt-oss` package during the image build
- only the model snapshot is cache-aware; a cold Docker image build still needs network access

### Nemotron 3

- starts on `:8003`
- serves OpenAI chat completions
- uses `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`
- follows the official NVFP4 vLLM recipe with `TRITON_ATTN`, FP8 KV cache, and `gpu-memory-utilization=0.9`
- bundles the `super_v3` reasoning parser locally, so cached launches do not need extra downloads

## Memory Rule

Treat DGX Spark as a one-big-LLM box. It has 128 GB of unified memory, and the large text models in this repo are
not meant to stay resident together.

| Model      | Why it is large on Spark                                                             |
|------------|--------------------------------------------------------------------------------------|
| Qwen       | `Qwen3-Coder-Next-FP8` is an 80B MoE model, and this repo gives vLLM 75% of memory |
| Qwen3.6    | `Qwen3.6-27B-FP8` weights are ~27 GB; KV cache eats the rest. Default budget 80%. |
| GPT-OSS    | NVIDIA recommends at least 70 GB free for `gpt-oss:120b`                            |
| Nemotron 3 | The official NVFP4 recipe already budgets 90% GPU memory and the Spark vLLM matrix lists 120B Nemotron support |

The deploy script removes peer containers before startup for exactly this reason.

## Offline Behavior

- Qwen and Nemotron look for cached snapshots under `~/.cache/huggingface/hub/models--*`.
- If the snapshot exists, they default to `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- If it does not exist, they keep online access enabled so the first start can download the model.
- GPT-OSS downloads into the mounted Hugging Face cache only when the pinned snapshot is missing or incomplete, then
  switches back to offline mode.
- GPT-OSS is not fully air-gapped unless the `gpt-oss-custom:latest` image already exists or Docker can reuse its
  existing build cache.

## Tuning Variables

Use env overrides instead of editing the launchers for each performance pass:

- Qwen: `QWEN_GPU_MEMORY_UTILIZATION`, `QWEN_MAX_NUM_SEQS`, `QWEN_MAX_NUM_BATCHED_TOKENS`,
  `QWEN_BLOCK_SIZE`, `QWEN_MAX_MODEL_LEN`, `QWEN_ATTENTION_BACKEND`
- GPT-OSS: `FORCE_REBUILD`, `ENABLE_NOTIFICATIONS`, `DEBUG_STARTUP`
  These are operational controls; the packaged vLLM settings still come from `Dockerfile.gpt-oss`.
- Nemotron: `NEMOTRON_GPU_MEMORY_UTILIZATION`, `NEMOTRON_MAX_NUM_BATCHED_TOKENS`,
  `NEMOTRON_MAX_NUM_SEQS`, `NEMOTRON_BLOCK_SIZE`, `NEMOTRON_MAX_MODEL_LEN`,
  `NEMOTRON_ATTENTION_BACKEND`
- Qwen3.6: `QWEN36_GPU_MEMORY_UTILIZATION`, `QWEN36_MAX_NUM_SEQS`,
  `QWEN36_MAX_NUM_BATCHED_TOKENS`, `QWEN36_BLOCK_SIZE`, `QWEN36_MAX_MODEL_LEN`,
  `QWEN36_ATTENTION_BACKEND`,
  `QWEN36_NUM_SPECULATIVE_TOKENS` (`0` off; `1` is the production-stable default — see README perf table),
  `QWEN36_KV_CACHE_DTYPE` (`auto`/`fp8`)

## Peer Cleanup

`deploy-model.sh` removes known peer model containers before start:

- `vllm_qwen_code`
- `vllm_qwen36`
- `gpt-oss`
- `vllm_nemotron3`
- `vllm_gemma4`
- `vllm_qwen_vl`
- `vllm_lfm`
- `flux_image`
- `comfyui_spark`

That keeps port usage and VRAM contention predictable on a reused Spark.

## Old Services

The deploy script warns if it finds these running systemd units:

- `qwen-server`
- `qwen3-vllm`
- `qwen3-coder-server`
- `gpt-oss-server`

If one of them is enabled, it can respawn a container after the deploy script removes it.

## Speculative Decoding (Qwen3.6 MTP)

Qwen3.6 ships native multi-token-prediction heads, so speculative decoding does not need a separate
draft model. `run-qwen36.sh` translates `QWEN36_NUM_SPECULATIVE_TOKENS` into:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":N}'
```

Measured on `spark-05` with vLLM `0.17.1.dev0` (NVIDIA GB10):

- `N=1` is stable across `c=1,4,8` and gives roughly +63% single-stream decode tps.
- `N=2` is fastest at low concurrency but the engine hits `cudaErrorIllegalAddress` at `c=8`.
- `N=3` crashes even at `c=1` mid-run. Not safe.

Production picks `N=1`. Higher values are useful for offline single-stream experiments.
Re-evaluate when the vLLM image is upgraded — the crash is a vLLM/MTP bug on this build, not a fundamental Qwen3.6 limit.

## Compact KV Cache

`QWEN36_KV_CACHE_DTYPE=fp8` is wired up but, on the current image and at depths up to 32 k,
did not change tps within noise. It is a memory knob (cuts KV cache footprint roughly in half),
not a speed knob. Reach for it when long-context (>64 k) deployments run out of headroom.

**TurboQuant KV-cache compression** ([ICLR 2026](https://arxiv.org/abs/2504.19874)) is
**not currently usable with vLLM on Spark**:

- The PyPI `turboquant 0.2.0` package targets HuggingFace `model.generate(past_key_values=...)`,
  which vLLM bypasses entirely.
- The community vLLM-integration fork (`0xSero/turboquant`) requires vLLM `0.18.0`; the pinned image is `0.17.1.dev0`.
- The fork's monkey-patch explicitly skips Mamba/linear-attention layers and warns that
  2-bit values cause `cos_sim ≈ 0.94` quality drift.

Re-attempt once the image moves to vLLM `0.18+` and only after measuring quality regression
on Qwen3.6 outputs.

## Benchmarking

`bench-qwen36.sh` runs `llama-benchy` against the local endpoint:

- `decode` profile — single-stream (`c=1`, `pp=512`, `tg=256`)
- `throughput` profile — concurrency `1, 4, 8` (`pp=1024`, `tg=256`)
- `longctx` profile — depth `0, 8192, 32768` (`pp=512`, `tg=32`)
- `all` runs the three sequentially

Run on the Spark itself; LAN latency inflates `ttfr` and `e2e_ttft`.
Outputs land under `~/spark-setup/perf-runs/qwen36-bench-<label>/`.

`llama-benchy` defaults to downloading `gpt2` for token counting. The harness sets
`HF_HUB_OFFLINE=1` and passes `--tokenizer "$MODEL"` so it reuses the served model's cached tokenizer.
