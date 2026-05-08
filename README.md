# spark-setup

Compact setup and deploy scripts for running vLLM-based models on NVIDIA DGX Spark.

This repository is intentionally narrow:

- Qwen3-Coder-Next on vLLM
- Qwen3.6-27B-FP8 on vLLM (with optional MTP speculative decoding)
- GPT-OSS 120B with the Responses API
- Nemotron 3 Super 120B NVFP4 on vLLM
- simple remote deploy wrappers
- smoke-test scripts for each model
- llama-benchy harness for repeatable perf comparisons

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

./deploy-model.sh <qwen|qwen36|gpt-oss|nemotron3|gemma4>
```

Run exactly one large model at a time. DGX Spark has 128 GB of unified memory, so these deployments are mutually
exclusive in practice.

Model-specific wrappers forward the same flags:

```bash
./deploy-qwen3.sh --dry-run
./deploy-qwen36.sh --start-only
./deploy-gpt-oss.sh --copy-only
./deploy-nemotron3.sh --start-only
```

## Models

| Deploy key  | Default model                                     | Port | Params / format            | Spark memory note                                      | Sources                           |
|-------------|---------------------------------------------------|------|----------------------------|--------------------------------------------------------|-----------------------------------|
| `qwen`      | `Qwen/Qwen3-Coder-Next-FP8`                       | 8000 | 80B total, 3B active, FP8 | Repo caps vLLM at 75% of Spark memory, about 96 GB     | [1](#sources), [2](#sources), [3](#sources) |
| `qwen36`    | `Qwen/Qwen3.6-27B-FP8`                            | 8005 | 27B dense, FP8            | 0.80 memory cap; weights ~27 GB. Optional MTP speculative decoding via `QWEN36_NUM_SPECULATIVE_TOKENS=1` (+63% decode tps on GB10) | [1](#sources), [9](#sources), [10](#sources) |
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

# Qwen3.6-27B-FP8 (MTP speculative decoding + KV cache toggle)
QWEN36_NUM_SPECULATIVE_TOKENS=1 \
QWEN36_GPU_MEMORY_UTILIZATION=0.80 \
QWEN36_MAX_MODEL_LEN=65536 \
QWEN36_MAX_NUM_SEQS=32 \
QWEN36_KV_CACHE_DTYPE=auto \
./deploy-qwen36.sh
```

### Qwen3.6 MTP behavior (measured on GB10, vLLM 0.17.1.dev)

| `QWEN36_NUM_SPECULATIVE_TOKENS` | Single-stream tg | Aggregate tg @ c=8 | Stability       |
|---------------------------------|------------------|--------------------|-----------------|
| `0` (off)                       | 7.75 t/s         | 57.5 t/s           | rock-solid      |
| `1`                             | **12.6 t/s**     | **87.2 t/s**       | **rock-solid**  |
| `2`                             | 14.1 t/s         | crashes at c=8     | unstable        |
| `3`                             | 14.2 t/s peak 20 | crashes at c=1     | unstable        |

`1` is the production-stable winner. Higher values give more peak speed but the engine hits `cudaErrorIllegalAddress` under load — see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

`QWEN36_KV_CACHE_DTYPE=fp8` is wired up but did **not** improve tps at depths up to 32k on this image; treat it as a memory knob, not a speed knob.

### Bare-metal Qwen3.6 (vLLM 0.20.1, optional)

For 200k+ contexts the Docker image (`vLLM 0.17.1.dev`) is too old. `run-qwen36-bare.sh` runs vLLM 0.20.1 directly from a venv on the Spark, on **port 8006** (so it can run alongside the Docker version on `:8005`):

```bash
# One-time install on the Spark
ssh spark-05
sudo apt-get install -y python3.12-dev build-essential ninja-build
python3 -m venv ~/spark-setup-baremetal/.venv
~/spark-setup-baremetal/.venv/bin/pip install -U vllm==0.20.1

# Launch — defaults to MTP=1, full 256k context, FP8 KV cache
QWEN36_BARE_NUM_SPECULATIVE_TOKENS=1 \
QWEN36_BARE_KV_CACHE_DTYPE=fp8 \
QWEN36_BARE_MAX_MODEL_LEN=262144 \
QWEN36_BARE_GPU_MEMORY_UTILIZATION=0.85 \
bash run-qwen36-bare.sh
```

Long-context single-stream numbers (Spark GB10, vLLM 0.20.1, MTP=1, FP8 KV):

| Context depth | Prefill t/s | Decode tg t/s | TTFT       |
|--------------:|------------:|--------------:|-----------:|
|             0 |        1223 |    14.46      |   0.4 s    |
|           64k |         753 |    14.09      |    88 s    |
|          128k |         624 |    13.77      |   211 s    |
|        **200k** |     **531** | **13.48**   | **378 s**  |

Decode rate degrades only **−7% from d=0 to d=200k** thanks to FP8 KV. Prefill drops sub-linearly (expected). At d=200k a fresh prompt takes ~6.3 minutes one-shot; once cached, decode runs at ~13.5 t/s per request. Pre-flight: install `python3.12-dev`, `build-essential`, and `ninja-build` (FlashInfer JIT-compiles attention kernels on first request).

For repeated tuning passes, log each run instead of editing shell history:

```bash
REMOTE=jetbrains@spark-07 \
QWEN_GPU_MEMORY_UTILIZATION=0.78 \
QWEN_MAX_NUM_SEQS=96 \
./perf-iteration.sh qwen iter-01
```

## Benchmarking with llama-benchy

`bench-qwen36.sh` wraps three `llama-benchy` profiles for repeatable comparisons:

| Profile      | What it measures                                              |
|--------------|---------------------------------------------------------------|
| `decode`     | single-stream `tg256` at `pp=512, c=1, d=0` (interactive)     |
| `throughput` | aggregate t/s at concurrency `1, 4, 8` (server load)          |
| `longctx`    | TTFT and decode at depths `0, 8192, 32768` (repo-level prompts) |

Run on the Spark itself (LAN latency distorts TTFT):

```bash
ssh spark-05
cd ~/spark-setup
uv pip install -U --system llama-benchy   # or: pip install --user --break-system-packages -U llama-benchy
bash bench-qwen36.sh all baseline          # writes perf-runs/qwen36-bench-baseline/
bash bench-qwen36.sh decode mtp1           # single profile, custom label
```

Results land under `~/spark-setup/perf-runs/qwen36-bench-<label>/{decode,throughput,longctx}.log`.

## Caches and Cold Starts

- `run-qwen3.sh` and `run-nemotron3.sh` inspect `~/.cache/huggingface` first.
- If weights are already cached, they start with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- If the cache is empty, they allow a first-run download automatically.
- GPT-OSS builds a custom image before it starts the model server.
- GPT-OSS follows the same rule for the model snapshot inside the container: if the pinned snapshot is missing or
  incomplete, it repairs the cache once, then forces offline mode.
- A fully offline GPT-OSS restart still needs a prebuilt `gpt-oss-custom:latest` image or warm Docker build layers.
- The Nemotron reasoning parser is bundled in this repository, so a cached Nemotron launch does not need a second
  network fetch for helper code.

## Repository Layout

- `deploy-model.sh`: shared remote deploy entrypoint
- `deploy-*.sh`: model-specific wrappers
- `hf-cache.sh`: shared Hugging Face cache probe helper
- `perf-iteration.sh`: logs one tuning pass under `perf-runs/`
- `run-*.sh`: remote container launchers (Docker)
- `run-qwen36-bare.sh`: bare-metal vLLM 0.20.1 launcher for 200k+ contexts (no Docker)
- `bench-qwen36.sh`: llama-benchy harness with `decode` / `throughput` / `longctx` / `huge` profiles
- `tq-serve.py`: TurboQuant + vLLM wrapper (currently non-functional — see `docs/README.md`)
- `*-load.sh`: smoke tests
- `Dockerfile.gpt-oss` and `in-container.sh`: GPT-OSS image build and runtime entrypoint
- [`docs/README.md`](docs/README.md): deployment notes and behavior differences
- [`docs/perf-log.md`](docs/perf-log.md): full experiment log — every config, every measurement, every decision
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
9. [Qwen3.6-27B-FP8 model card](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)
10. [vLLM recipe for Qwen3.6-27B (MTP, reasoning parser)](https://recipes.vllm.ai/Qwen/Qwen3.6-27B)
