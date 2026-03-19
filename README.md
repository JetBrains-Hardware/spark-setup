# spark-setup

Compact setup and deploy scripts for running vLLM-based models on NVIDIA DGX Spark.

This repository is intentionally narrow:

- Qwen3-Coder-Next on vLLM
- GPT-OSS 120B with the Responses API
- Nemotron 3 8B on vLLM
- simple remote deploy wrappers
- smoke-test scripts for each model

It does not include private demo code, dashboard apps, or internal JetBrains integrations.

## Prerequisites

- A reachable DGX Spark host over SSH
- Docker on the Spark
- Enough free disk for model weights and images
- A Hugging Face token for authenticated downloads
- `REMOTE=user@host` exported on the machine running the deploy scripts

## Quick Start

```bash
export REMOTE=jetbrains@your-spark-host
export HF_TOKEN=hf_...

./deploy-model.sh qwen
./deploy-model.sh gpt-oss
./deploy-model.sh nemotron3
```

Model-specific wrappers forward the same flags:

```bash
./deploy-qwen3.sh --dry-run
./deploy-gpt-oss.sh --copy-only
./deploy-nemotron3.sh --start-only
```

## Models

| Model       | Script              | Port | API                         |
|-------------|---------------------|------|-----------------------------|
| `qwen`      | `run-qwen3.sh`      | 8000 | OpenAI chat completions     |
| `gpt-oss`   | `run-gpt-oss.sh`    | 8001 | OpenAI Responses API        |
| `nemotron3` | `run-nemotron3.sh`  | 8003 | OpenAI chat completions     |

The deploy flow copies the minimum files for the selected model, stops peer model containers, waits for health,
and runs a smoke test on the remote Spark.

## Common Flags

```bash
./deploy-model.sh <qwen|gpt-oss|nemotron3> --dry-run
./deploy-model.sh <qwen|gpt-oss|nemotron3> --copy-only
./deploy-model.sh <qwen|gpt-oss|nemotron3> --start-only
```

## Caches and Cold Starts

- Qwen deploys force online mode for the initial startup path so a cold cache can download weights.
- Direct `run-qwen3.sh` stays offline-first unless you override `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`.
- GPT-OSS builds a custom image before it starts the model server.
- Nemotron 3 uses a regular vLLM image and typically has the smallest startup footprint of the three.

## Repository Layout

- `deploy-model.sh`: shared remote deploy entrypoint
- `deploy-*.sh`: model-specific wrappers
- `run-*.sh`: remote container launchers
- `*-load.sh`: smoke tests
- `Dockerfile.gpt-oss` and `in-container.sh`: GPT-OSS image and runtime patching
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
