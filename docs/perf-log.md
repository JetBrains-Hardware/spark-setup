# Qwen3.6-27B-FP8 on DGX Spark — Experiment Log

Single source of truth for every config we tested, the numbers we got, and the decision that came out of each phase. Newest at the bottom.

## Hardware & host

- **Host**: `spark-05` (DGX Spark, GB10 — Grace + Blackwell, sm_121, 128 GiB unified memory)
- **OS**: Ubuntu 24.04.4 LTS aarch64, kernel `6.17.0-1014-nvidia`, glibc 2.39
- **NVIDIA driver**: 580.142, CUDA runtime 13.1.1
- **Python**: 3.12.3
- **Model**: `Qwen/Qwen3.6-27B-FP8` (dense, FP8 weights, ~27 GB on disk)
  - 64 layers, GQA 24/4, head_dim 256, max_position_embeddings 262144
  - `model_type=qwen3_5` (Qwen3.6 inherits Qwen3.5 architecture)

## Benchmark methodology

- Tool: `llama-benchy 0.3.7` against the OpenAI-compatible vLLM endpoint.
- Run **on the Spark itself** (LAN/Tailscale latency distorts TTFR/TTFT).
- Profiles defined in `bench-qwen36.sh`:
  - `decode` — single-stream `tg256` at `pp=512, c=1, d=0` (interactive decode).
  - `throughput` — aggregate at concurrency `1, 4, 8`, `pp=1024, tg=256`.
  - `longctx` — depth `0, 8192, 32768`, `pp=512, tg=32`, `c=1`.
  - `huge` — depth `0, 65536, 131072, 200000`, `pp=256, tg=16`, `c=1`.
- 3 runs per cell (2 for `huge`), mean ± stddev reported.
- Tokenizer forced to the served model's own to avoid llama-benchy's default `gpt2` download.

---

## Phase 1 — Docker MTP sweep (vLLM 0.17.1.dev0, FP16 KV, max_model_len 65536)

Image: `scitrera/dgx-spark-vllm:0.17.0-t5`. Container `vllm_qwen36` on `:8005`.

| `QWEN36_NUM_SPECULATIVE_TOKENS` | Decode tg c=1 (t/s) | Peak (t/s) | TG total c=8 (t/s) | Stability |
|---:|---:|---:|---:|:---|
| `0` (off) | 7.75 ± 0.02 | 8.00 | 57.5 | rock-solid |
| `1` | **12.63 ± 0.31** | 14.33 | **87.2** | **rock-solid** |
| `2` | 14.13 ± 0.57 | 18.67 | engine crash @ c=8 (`cudaErrorIllegalAddress`) | unstable |
| `3` | 14.22 ± 0.00 | 20.00 | engine crash even @ c=1 mid-run | unstable |

**Decision**: production picks `MTP=1`. Higher values give peak gains but crash under load on this image's MTP path.

Long-context, MTP=1 (`longctx` profile, c=1):

| Depth | tg (t/s) | TTFR pp512 (ms) |
|---:|---:|---:|
| 0 | 12.81 ± 0.38 | 462 |
| 8192 | 13.12 ± 0.36 | 9183 |
| 32768 | 11.50 ± 0.25 | 40862 |

## Phase 2 — Docker MTP=1 + `--kv-cache-dtype fp8`

Same image, same container, `QWEN36_KV_CACHE_DTYPE=fp8`.

| Test | MTP=1 (FP16 KV) | MTP=1 + KV=fp8 | Δ |
|:---|---:|---:|---:|
| decode tg256 c=1 | 12.63 | 12.18 | −3.6% |
| tg256 c=8 total | 87.20 | 87.40 | within noise |
| longctx d=8192 tg | 13.12 | 12.60 | −4.0% |
| longctx d=32768 tg | 11.50 | 11.80 | +2.6% |

**Decision**: at depths ≤ 32k FP8 KV is **a memory knob, not a speed knob** — within noise. Useful only when KV memory is the binding constraint (long context), confirmed below in Phase 4.

## Phase 3 — TurboQuant for vLLM (attempt, failed)

The user's original ask was "compact KV cache via TurboQuant". Tried `0xSero/turboquant` (the only public vLLM-integration fork of TurboQuant ICLR 2026):

- PyPI `turboquant 0.2.0` is a HuggingFace `model.generate()` package — vLLM bypasses that path entirely.
- `0xSero/turboquant` GitHub fork has no releases, last tested on Qwen3.5; we patched two real bugs in place:
  1. `import os` missing in `turboquant/vllm_attn_backend.py` (used in `patched_worker_load`).
  2. `patched_layout_update(self, kv_caches)` signature outdated — vLLM 0.20.1's `_update_hybrid_attention_mamba_layout` now takes `(kv_caches, kernel_block_sizes)`. Patched to `*args, **kwargs`.
- After patches, hit a hard wall:
  ```
  RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture
  unless the CPU tensor is pinned.
  ```
  Fork's hot path does an unpinned CPU↔CUDA copy that vLLM 0.20's CUDA-graph capture forbids.

**Decision**: TurboQuant **not feasible on this stack today**. `tq-serve.py` is committed for future retries when the fork's tensor allocations are fixed (or a new fork lands). Use vLLM's built-in `--kv-cache-dtype fp8` as the practical compact-KV substitute (Phase 4).

## Phase 4 — Bare-metal vLLM 0.20.1 + MTP=1 + FP8 KV + max_model_len 262144

Switched off Docker's `0.17.1.dev0` to bare metal so we could run modern vLLM and reach native 256 k context. Venv at `~/spark-setup-baremetal/.venv`.

Pre-flight on Spark:
```bash
sudo apt-get install -y python3.12-dev build-essential ninja-build
python3 -m venv ~/spark-setup-baremetal/.venv
~/spark-setup-baremetal/.venv/bin/pip install -U vllm==0.20.1
```

**Why those packages**: `python3.12-dev` and `build-essential` because `fastsafetensors` ships only an aarch64 sdist. `ninja-build` because FlashInfer JIT-compiles attention kernels on first inference.

Launcher `run-qwen36-bare.sh` on `:8006` (parallel to Docker on `:8005`).

Parity vs Docker (MTP=1):

| Test | Docker 0.17.1 (FP16 KV, 64k) | Bare 0.20.1 (FP16 KV, 64k) | Δ |
|:---|---:|---:|---:|
| decode tg256 c=1 | 12.63 | 12.84 | +1.7% |
| tg c=8 total | 87.20 | 88.99 | +2.1% |
| longctx d=32768 tg | 11.50 | 11.63 | +1.1% |

Bare-metal vLLM 0.20.1 is **at parity or +2-3% faster** than the Docker 0.17.1.dev image at MTP=1.

Production config (`MTP=1, KV=fp8, max-model-len=262144`):

| Test | Docker 0.17 MTP=1 | Bare 0.20 MTP=1 + KV=fp8 + 256k | Δ |
|:---|---:|---:|---:|
| decode tg256 c=1 | 12.63 | 12.38 | −2.0% |
| tg c=8 total | 87.20 | 87.97 | +0.9% |
| longctx d=32768 tg | 11.50 | 13.15 | **+13.5%** |

At long context the FP8 KV win compounds with the bare-metal vLLM jump.

### Phase 4b — `huge` profile (200k+ context, the user's stated goal)

Bare-metal, MTP=1, KV=fp8, max-model-len=262144, single stream:

| Depth | Prefill t/s | Decode tg t/s | TTFT |
|---:|---:|---:|---:|
| 0 | 1223 ± 7 | 14.46 ± 0.00 (peak 15) | 0.4 s |
| 64 k | 753 ± 0 | 14.09 ± 0.01 | 88 s |
| 128 k | 624 ± 4 | 13.77 ± 0.11 | 211 s |
| **200 k** | **531 ± 2** | **13.48 ± 0.11** | **378 s** |

Decode rate degrades **−7% from d=0 to d=200 k** thanks to FP8 KV. Prefill drops sub-linearly (expected). At d=200 k a fresh prompt takes ~6.3 minutes one-shot; once cached, decode runs at ~13.5 t/s per request.

**Decision**: `MTP=1 + KV=fp8 + max-model-len=262144` is the bare-metal production config. Logged numbers are the baseline for any further perf experiment.

## Phase 5 — DFlash speculative decoding (in progress)

The user's "DTree" maps to **DDTree** ([Technion, ICLR-track 2026](https://liranringel.github.io/ddtree/)) — a draft-tree built on top of DFlash. DDTree is **not yet upstream in vLLM**; only DFlash is, in [PR #40898](https://github.com/vllm-project/vllm/pull/40898). Plan: ship DFlash now, swap in DDTree once it lands.

Reported numbers (Qwen3-Coder-30B-A3B HumanEval, T=0): **DFlash 6.09× / DDTree 8.22×** lossless speedup over baseline.

### Setup attempts on this Spark

| Attempt | What we did | Outcome |
|:---|:---|:---|
| 1 | `pip install vllm @ git+...PR#40898` into a separate `~/spark-setup-baremetal/dflash/.venv` | Built. `import vllm` shows `0.19.2rc1.dev129+g3cfc8f8b7.cu130`. |
| 2 | First launch (`--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash"}'`) | `OSError: gated repo` — `z-lab/Qwen3.6-27B-DFlash` requires HF access grant. |
| 3 | Mirrored `~/.cache/huggingface/token` to Spark and retried | Still 401. Local token at `~/.cache/huggingface/token` was **expired** (March-19 file). |
| 4 | Switched to fresh token at `~/.hf`, mirrored to Spark | Token validates; access grant **active** (HTTP 200 on the gated config.json). |
| 5 | Launch with FP8 base + DFlash drafter | Engine init crashed: `RuntimeError: cutlass_scaled_mm ... NotImplementedError: No compiled cutlass_scaled_mm for a compute capability less than CUDA device capability: 121`. The PR's wheel was compiled without sm_121 in `TORCH_CUDA_ARCH_LIST`. |
| 6 | Source-rebuild attempt #1 (`--no-build-isolation`) | `ModuleNotFoundError: setuptools_scm`. Installed it. |
| 7 | Rebuild #2 | `ModuleNotFoundError: pybind11` (transitive `fastsafetensors` build dep). Installed it. |
| 8 | Rebuild #1 (`--no-build-isolation`) | Failed: `ModuleNotFoundError: setuptools_scm`. |
| 9 | Pre-installed `setuptools_scm`, rebuild #2 | Failed: `ModuleNotFoundError: pybind11` (transitive `fastsafetensors` build dep). |
| 10 | Pre-installed `pybind11`, rebuild #3/#4 | Compiled most kernels for sm_121, then failed at `nvfp4_kv_cache_kernels.cu`: `ptxas error: Instruction 'cvt with .e2m1x2' not supported on .target 'sm_121'`. The `cvt.e2m1x2` (NVFP4 conversion) PTX is sm_120-only; vLLM's `cuda_archs_loose_intersection` family-fallback for `12.0f` was matching sm_121. |
| 11 | Patched `vllm-src/CMakeLists.txt` to drop `12.1a` from the SM120 FP4 path and replace `12.0f` with `12.0` (no family fallback). Now rebuilding from local patched source. | In flight at the time of writing. |

### Open decisions

- If rebuild #3 lands and DFlash runs on FP8 base → bench `decode` and `huge`, log here.
- If rebuild #3 fails on another sm_121 kernel → either pin to BF16 base (`Qwen/Qwen3.6-27B`, ~54 GB) which doesn't hit the FP8 cutlass paths, or wait for PR #40898 to merge into stable.
- DDTree integration is **fork-not-upstream**. Re-evaluate once a vLLM-merged version exists. Don't run a custom DDTree fork until upstream stabilises.

## Phase 6 — SFP+ inter-host network (open, blocked)

Three hosts in the intended mesh: `spark-05`, `spark-07`, `thor-04`. Audit (May 7):

- `spark-05` and `spark-07` each have **4× `mlx5_core` ports** (Mellanox ConnectX). All show `carrier=0`, `Link detected: no`. Some report "Direct Attach Copper, No partner detected" — DAC plugged in, no partner.
- `thor-04` is offline (ssh times out on LAN, Tailscale shows peer with no active connection).

**Blockers**:
1. Physical-layer issue on the SFP+ ports — `ip link set up` won't help; the cables aren't carrying signal.
2. `thor-04` powered off — can't be part of any topology.

**Decision**: park until thor-04 is on, cabling is verified, and the topology (direct mesh vs. switch) is chosen. Then assign a `10.10.0.0/24` subnet via netplan and validate with `iperf3`.
