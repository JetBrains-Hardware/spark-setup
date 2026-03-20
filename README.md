# spark-setup

Compact setup and deploy scripts for running vLLM-based models on NVIDIA DGX Spark.

This repository is intentionally narrow:

- Qwen3-Coder-Next on vLLM
- GPT-OSS 120B with the Responses API
- Nemotron 3 Super 120B NVFP4 on vLLM
- simple remote deploy wrappers
- smoke-test scripts for each model

It does not include private demo code, dashboard apps, or internal JetBrains integrations.

## Prerequisites

- A reachable DGX Spark host over SSH
- Docker on the Spark
- Enough free disk for model weights and images
- A Hugging Face token for the first authenticated download
- `REMOTE=user@host` exported on the machine running the deploy scripts

## Quick Start

```bash
export REMOTE=jetbrains@your-spark-host
export HF_TOKEN=hf_...

./deploy-model.sh <qwen|gpt-oss|nemotron3>
```

Run exactly one large model at a time. DGX Spark has 128 GB of unified memory, so these deployments are mutually
exclusive in practice.

Model-specific wrappers forward the same flags:

```bash
./deploy-qwen3.sh --dry-run
./deploy-gpt-oss.sh --copy-only
./deploy-nemotron3.sh --start-only
```

## Models

| Deploy key  | Default model                                     | Port | Params / format            | Spark memory note                                      | Sources                           |
|-------------|---------------------------------------------------|------|----------------------------|--------------------------------------------------------|-----------------------------------|
| `qwen`      | `Qwen/Qwen3-Coder-Next-FP8`                       | 8000 | 80B total, 3B active, FP8 | Repo caps vLLM at 75% of Spark memory, about 96 GB     | [1](#sources), [2](#sources), [3](#sources) |
| `gpt-oss`   | `openai/gpt-oss-120b`                             | 8001 | 117B total, 5.1B active, MXFP4 | NVIDIA recommends 70 GB free for the 120B path     | [1](#sources), [4](#sources), [5](#sources) |
| `nemotron3` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | 8003 | 120B total, 12B active, NVFP4 | Official recipe uses 90% memory; Spark vLLM docs list 120B Nemotron support | [1](#sources), [6](#sources), [7](#sources), [8](#sources) |

The deploy flow copies the minimum files for the selected model, stops peer model containers, waits for health,
and runs a smoke test on the remote Spark.

This repo uses the Nemotron NVFP4 checkpoint because NVIDIA publishes a direct vLLM recipe for it with
Spark-friendly settings such as `--kv-cache-dtype fp8`, `--attention-backend TRITON_ATTN`, and
`--gpu-memory-utilization 0.9`.

Do not keep two of these models resident at the same time. Once one 80B+ model and its KV cache are loaded, the
remaining headroom is not enough for another large text model.

## Common Flags

```bash
./deploy-model.sh <qwen|gpt-oss|nemotron3> --dry-run
./deploy-model.sh <qwen|gpt-oss|nemotron3> --copy-only
./deploy-model.sh <qwen|gpt-oss|nemotron3> --start-only
```

## Performance Knobs

Qwen and Nemotron expose their main runtime tuning knobs as environment variables, so repeated performance passes
do not need script edits. GPT-OSS exposes deploy-time build/debug controls here, but its actual vLLM patch settings
still live in `Dockerfile.gpt-oss`.

```bash
# Qwen
QWEN_GPU_MEMORY_UTILIZATION=0.78 \
QWEN_MAX_NUM_SEQS=96 \
QWEN_MAX_NUM_BATCHED_TOKENS=12288 \
./deploy-qwen3.sh

# GPT-OSS
FORCE_REBUILD=1 \
ENABLE_NOTIFICATIONS=0 \
./deploy-gpt-oss.sh

# Nemotron
NEMOTRON_GPU_MEMORY_UTILIZATION=0.88 \
NEMOTRON_MAX_NUM_BATCHED_TOKENS=12288 \
NEMOTRON_MAX_NUM_SEQS=384 \
./deploy-nemotron3.sh
```

For repeated tuning passes, log each run instead of editing shell history:

```bash
REMOTE=jetbrains@spark-07 \
QWEN_GPU_MEMORY_UTILIZATION=0.78 \
QWEN_MAX_NUM_SEQS=96 \
./perf-iteration.sh qwen iter-01
```

## Caches and Cold Starts

- `run-qwen3.sh` and `run-nemotron3.sh` inspect `~/.cache/huggingface` first.
- If weights are already cached, they start with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- If the cache is empty, they allow a first-run download automatically.
- GPT-OSS builds a custom image before it starts the model server.
- GPT-OSS follows the same rule for the model snapshot inside the container: download once, then force offline mode.
- A fully offline GPT-OSS restart still needs a prebuilt `gpt-oss-custom:latest` image or warm Docker build layers.
- The Nemotron reasoning parser is bundled in this repository, so a cached Nemotron launch does not need a second
  network fetch for helper code.

## Repository Layout

- `deploy-model.sh`: shared remote deploy entrypoint
- `deploy-*.sh`: model-specific wrappers
- `hf-cache.sh`: shared Hugging Face cache probe helper
- `perf-iteration.sh`: logs one tuning pass under `perf-runs/`
- `run-*.sh`: remote container launchers
- `*-load.sh`: smoke tests
- `Dockerfile.gpt-oss` and `in-container.sh`: GPT-OSS image build and runtime entrypoint
- [`docs/README.md`](docs/README.md): deployment notes and behavior differences
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md): common failure modes
- [`examples/README.md`](examples/README.md): minimal config examples

## Validation

Local checks:

```bash
bash -n deploy-model.sh deploy-qwen3.sh deploy-gpt-oss.sh deploy-nemotron3.sh \
  run-qwen3.sh run-gpt-oss.sh run-nemotron3.sh \
  qwen3-load.sh gpt-oss-load.sh nemotron3-load.sh in-container.sh

git diff --check
```

Optional dry-run checks:

```bash
REMOTE=user@host ./deploy-qwen3.sh --dry-run
REMOTE=user@host ./deploy-gpt-oss.sh --dry-run
REMOTE=user@host ./deploy-nemotron3.sh --dry-run
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

## Sources

1. [DGX Spark hardware overview: 128 GB unified memory and support for models up to 200B](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
2. [Qwen3-Coder-Next-FP8 model card](https://huggingface.co/Qwen/Qwen3-Coder-Next-FP8)
3. [Qwen launcher budget in this repo](run-qwen3.sh)
4. [gpt-oss-120b model card](https://huggingface.co/openai/gpt-oss-120b)
5. [NVIDIA Spark OpenShell guidance: keep at least 70 GB free for `gpt-oss:120b`](https://build.nvidia.com/spark/openshell)
6. [NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)
7. [Bundled `super_v3` reasoning parser used by the Nemotron launcher](super_v3_reasoning_parser.py)
8. [NVIDIA DGX Spark `vLLM for Inference` support matrix](https://build.nvidia.com/spark/vllm)
