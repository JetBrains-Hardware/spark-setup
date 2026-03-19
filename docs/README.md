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
- deploy path temporarily enables online Hugging Face access for cold-cache downloads
- tuned for the larger coder workload and longer context

### GPT-OSS

- starts on `:8001`
- serves the OpenAI Responses API
- builds a custom container image before startup
- patches the installed `gpt-oss` package at container startup via `in-container.sh`

### Nemotron 3

- starts on `:8003`
- serves OpenAI chat completions
- uses a lighter-weight vLLM startup path than Qwen or GPT-OSS

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
