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

## Peer Cleanup

`deploy-model.sh` removes known peer model containers before start:

- `vllm_qwen_code`
- `gpt-oss`
- `vllm_nemotron3`
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
