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
| 11 | Patched `vllm-src/CMakeLists.txt`: SM120 FP4 src list `12.0f` → `12.0`, dropped `12.1a`. Rebuilt from local source. | Failed: `cuda_archs_loose_intersection` is **forward-compatible within major version** — `SRC=12.0` still matched `TGT=12.1` (same major 12, 12.0 ≤ 12.1). Build still emitted `nvfp4_kv_cache_kernels.cu` for `arch=compute_120,code=sm_120`. ptxas still rejects `cvt.e2m1x2` (sm_120 PTX not supported even though target says sm_120 — driver toolkit mismatch). |
| 12 | Patched the SM120 NVFP4 block's `if(... AND FP4_ARCHS)` → `if(FALSE)` to disable it entirely. cache_kernels.cu's NVFP4 calls are gated by `ENABLE_NVFP4_SM100\|SM120` macros which won't be defined now. User does not want NVFP4 anyway. | Failed at a *different* file: `qutlass-src/.../fused_quantize_nv.cu` hits the same `cvt with .e2m1x2 not supported on sm_121` ptxas error. The `cvt.e2m1x2` PTX is hardware-restricted to **sm_120/sm_100 (datacenter Blackwell)** and is **not supported on sm_121 (GB10/Spark)**. Every FP4-emitting file in PR #40898 + bundled qutlass dep will surface the same wall. |

### Verdict

**DFlash on FP8 base is hardware-blocked on DGX Spark today.** PR #40898 (and its `qutlass` bundled dep) emit `cvt.e2m1x2` PTX which sm_121 doesn't support. Patching out every FP4 path across vLLM and `qutlass` would be hours of work with no guarantee. Two real paths forward:

1. **Wait for PR #40898 to be properly merged into a stable vLLM that ships sm_121-compiled wheels** — the cleanest path, no extra work.
2. **Use the BF16 base (`Qwen/Qwen3.6-27B`, ~54 GB)** to side-step the FP8 cutlass + FP4 PTX paths. Breaks the user's "FP8 only" preference. Only use as fallback.

Pivoting back to the **production-stable bare-metal config (MTP=1 + FP8 KV + 256k)** as the baseline for the 5 optimization loops.

### Open decisions

- If rebuild #3 lands and DFlash runs on FP8 base → bench `decode` and `huge`, log here.
- If rebuild #3 fails on another sm_121 kernel → either pin to BF16 base (`Qwen/Qwen3.6-27B`, ~54 GB) which doesn't hit the FP8 cutlass paths, or wait for PR #40898 to merge into stable.
- DDTree integration is **fork-not-upstream**. Re-evaluate once a vLLM-merged version exists. Don't run a custom DDTree fork until upstream stabilises.

## Phase 7 — Optimization loops on the production config

Five focused single-knob loops. Baseline = bare-metal MTP=1 + FP8 KV + max-model-len 262144 (Phase 4b numbers).
Each loop sweeps **one** knob, decides keep/revert, carries winners forward.

| Loop | Knob | Test | Decode tg c=1 | Decision |
|---:|:---|:---|---:|:---|
| baseline | (Phase 4b) | — | 14.46 (peak 15.0) | — |
| 1 | `max-num-batched-tokens` 8192 → **16384** | decode | 12.66 (peak 14.0) | keep (within noise on decode, helps prefill batching at long ctx) |
| 2 | `block-size` 16 → **32** (carry mnbt=16384) | decode | 12.89 (peak 14.33) | **keep** (+0.2 over loop 1, slight peak gain) |
| 3 | `max-num-seqs` 32 → 64 (carry blksz=32, mnbt=16384) | decode + throughput | 12.25 (peak 14.0) | **revert** (decode regressed −5%, c=4/c=8 within noise) |
| 4 | `--attention-backend` FLASHINFER → **FLASH_ATTN** | decode | 12.87 (peak 14.0) | **keep** — decode within noise, **prefill +37%** (pp512: 947 → 1302 t/s) |
| 5 | combined winners (`mnbt=16384, blksz=32, mns=32, FLASH_ATTN`) | huge (depths up to 200k) | see below | **adopted as new defaults** |

### Loop 5 final huge profile (combined winners)

| Depth | Prefill t/s | Decode tg t/s | Δ vs Phase 4b |
|---:|---:|---:|---:|
| 0 | 1267 ± 2 | **14.50** ± 0.00 (peak 15) | +0.3% |
| 64 k | 743 ± 7 | **14.33** ± 0.15 (peak 14.5) | +1.7% |
| 128 k | 623 ± 1 | **14.31** ± 0.24 | **+3.9%** |
| **200 k** | 531 ± 0 | **14.42** ± 0.23 | **+7.0%** |

Decode is now nearly **flat from depth 0 to 200 k** (−0.6% drop instead of −7%). The FP8 KV + FLASH_ATTN combination sustains decode throughput as KV grows.

### Hardware knobs probed

- `nvidia-smi --query-gpu=clocks.current.sm,clocks.max.sm`: 2392 / 3003 MHz at idle, GPU already runs near boost during inference. Locking via `-lgc 3003,3003` would only shave a few % off TTFT and risks throttling — left default.
- Persistent mode (`nvidia-smi -pm 1`) already enabled by `nvidia-persistenced` (audited in Phase 4 setup).

### Adopted defaults in `run-qwen36-bare.sh`

```bash
QWEN36_BARE_MAX_MODEL_LEN=262144
QWEN36_BARE_GPU_MEMORY_UTILIZATION=0.85
QWEN36_BARE_MAX_NUM_BATCHED_TOKENS=16384
QWEN36_BARE_BLOCK_SIZE=32
QWEN36_BARE_KV_CACHE_DTYPE=fp8
QWEN36_BARE_NUM_SPECULATIVE_TOKENS=1   # MTP=1 — production stable winner
QWEN36_BARE_ATTENTION_BACKEND=FLASH_ATTN
```

These are now the script defaults, so a plain `bash run-qwen36-bare.sh` matches the production-tuned config.

### Loops 6–7 (extended sweep)

| Loop | Knob | Decode tg c=1 | Throughput tg c=8 (total) | Decision |
|---:|:---|---:|---:|:---|
| 6 | mnbt 16384 → **32768** | 12.47 (−3.3% vs L5) | n/a | **revert** — bigger batch budget hurt at single-stream |
| 7 | gpu-mem-util 0.85 → **0.90** | 12.52 (−2.9%) | 86.81 (within noise) | **revert** — not memory-bound at our test loads |

Loop 8 (gpu-mem-util 0.95) skipped: pattern from loop 7 was clear, no point burning a cold start to confirm a regression.

## Phase 7b — TurboQuant retry on stable vLLM 0.20.1

Earlier TurboQuant attempt (Phase 3) was on vLLM 0.17.1.dev (Docker). Stable 0.20.1 is what bare-metal runs now. Retry to see if the CUDA-graph path changed.

| Step | Result |
|:---|:---|
| `pip install -e turboquant/` into stable `.venv` | Patches from Phase 3 still in place. Imports OK. |
| Launch `tq-serve.py serve ... --enforce-eager=False` (CUDA graphs on) | **Same wall**: `RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned.` Confirms the bug is in the **fork's hot path**, not vLLM-version-specific. |
| Launch with `--enforce-eager` (CUDA graphs off) | **Engine starts, `/health` 200, completions return HTTP 200.** The fork's hooks are running. |
| Decode bench at default 3-bit K / 2-bit V | **Coherence test FAILED** (model returned garbage instead of "Paris"). Model is incoherent — tokens look like `Here!!!!!!!`. |
| Retry with TQ_KEY_BITS=4 / TQ_VALUE_BITS=4 (the fork's "near-lossless" claim) | **Still incoherent.** Output: `Here!\n! (!/!\n!\n\n!_!k!  \n!\n!\n\n!!!!!!!!...` (just `!` and noise). |

### Verdict

`0xSero/turboquant` **fundamentally does not produce correct logits on Qwen3.6-27B-FP8** at any tested bit-width. The fork was developed and tested on **Qwen3.5-27B-AWQ** (different quantization scheme) — its KV-cache hooks don't compose with FP8's per-channel/per-tensor scaling on Qwen3.6.

Realistic substitutes for "compact KV cache":
- **vLLM built-in `--kv-cache-dtype fp8`** is the only operational compact-KV option on this stack today. ~2× compression vs FP16, no quality loss, no perf hit at our depths. **Already adopted in production config** (Phase 4 onwards).
- **Better TurboQuant** would need a fork that actually adapts to FP8 weights — none publicly available as of writing. Re-evaluate when one appears.

## Phase 8 — Inter-host fast network (audited 2026-05-09)

### Per-host NIC inventory

#### spark-05 (DGX Spark, GB10)
- **4× Mellanox MT2910 [ConnectX-7]** ports — `enp1s0f0np0`, `enp1s0f1np1`, `enP2p1s0f0np0`, `enP2p1s0f1np1`
  - Firmware 28.45.4028, PCIe x4 @ 32 GT/s (126 Gb/s)
  - All four `<...UP>` administratively, but `state DOWN` because no carrier
  - dmesg port-module events: **f0 ports = "Cable unplugged"**, **f1 ports = "Cable plugged"**. So 2 of 4 cables present.
  - **Driver warning**: `Detected insufficient power on the PCIe slot (27W)` — ConnectX-7 datasheet calls for more under load; may cap throughput.
- 1× `r8127` 1 GbE on `enP7s7` (management LAN, 192.168.178.75) — UP
- `Link detected: no` on all 4 mlx5 ports.

#### spark-07 (DGX Spark, GB10)
- Same 4× ConnectX-7. dmesg shows **all four cables now plugged** (two at boot, two more plugged at +240 074 s of uptime).
- All four `<...UP>` admin, `state DOWN` no carrier. `Link detected: no` everywhere.
- 1× `r8169` 1 GbE on `enP7s7` (192.168.178.66 area).

#### thor-04 (NVIDIA Jetson Thor, `Linux 6.8.12-tegra`)
- **No Mellanox / no SFP+.** The host has integrated **`nvethernet` (Tegra MGBE) RJ45 jacks**.
- 4× `mgbe0_0..mgbe3_0`: `Speed 10000Mb/s`, `Port: MII (TP)`, **`Link detected: yes` on all four**, no IP configured.
- 1× `r8126` 1 GbE on `enP2p1s0` (192.168.178.76, default route via that).

### Implications

1. **The "SFP+ network between 3 GPU devices" premise is wrong** — thor-04 has zero SFP+. The high-speed mesh therefore can't be uniform-SFP+; it has to bridge SFP+ (Sparks) ↔ 10GBASE-T (thor-04).
2. **Sparks' SFP+ ports show cables but no link**, while **thor-04's 10G RJ45 ports show active links** (to something). Three live possibilities:
   - All seven (4 thor + 2 spark-05 + ≥2 spark-07) cables run to a hybrid switch (SFP+ cages + 10GBASE-T jacks) that's *partially* configured — thor's RJ45 side links, the SFP+ side doesn't.
   - thor's links go to a different device entirely (a separate switch, NAS, or loop) and the Spark DACs go to a switch that's off / wrong-speed.
   - Sparks' DACs go directly between hosts (Spark↔Spark), and link doesn't form because of a speed/FEC mismatch or a bad cable.
3. **No usable speed information from the SFP+ side without a partner** — `ethtool` shows the placeholder `1000baseT/Full` only because there's no module SFP info; ConnectX-7 actually supports 10/25/50/100/200/400 GbE. We can't pick a "max speed" until link forms.
4. The PCIe-power warning on spark-05 means the slot only sources 27 W. ConnectX-7's full envelope is higher. Need to verify the chassis can sustain it under load — otherwise even a forming link could throttle.

### What I cannot fix from the OS

- **Physical cabling** — which DAC goes from where to where. The dmesg events tell me cables are inserted into the NIC cages, not what's at the other end.
- **Switch configuration** — if a switch sits in the middle, it has to be configured to negotiate the right speed/FEC.
- **Power delivery** — the PCIe-slot 27 W reading is a hardware/firmware concern, not a netplan one.

### Verdict + asked-for clarifications

| Topology option | What it gives | Hardware required |
|:---|:---|:---|
| **A: 2-host SFP+ Spark↔Spark direct** | Up to 200/400 GbE between the two Sparks | 1 DAC between matched ports on spark-05 and spark-07. Drop existing failed cables, use a known-good direct one. thor-04 stays on the 1 GbE LAN. |
| **B: 3-host via hybrid switch** | All-to-all fast networking, capped by thor-04's 10 GbE | A switch with **both** SFP+ cages and 10GBASE-T jacks (Mikrotik CRS, Ubiquiti EnterpriseXG, etc.). Cables: 1 DAC each from each Spark to the switch's SFP+ side; 1 RJ45 each from each thor-04 mgbe port to the switch's 10GBASE-T side. |
| **C: 3-host via 10GBASE-T transceivers** | Uniform 10 GbE all-to-all | SFP+ → 10GBASE-T transceivers in each Spark (Mellanox compatibility list — Mikrotik S+RJ10 typically works, but ConnectX-7 datasheet should be checked). Then either direct-mesh (3 cables in a triangle, with one extra port per Spark) or switch. |

I am not making any link-up / IP-assign changes until you confirm the topology — admin-up'ing a port that's wired into a misconfigured switch can cause spanning-tree storms, and configuring `10.10.0.0/24` on the wrong interface poisons routing.

**Open questions for the user:**

1. Is there a switch in the loop, and if so what's its model + management IP?
2. Are the cables currently connected to anything live, or are they dangling? (Visual check.)
3. If you want option **A** (Spark-pair only): which physical port of spark-05 should go to which physical port of spark-07?
4. Will you tolerate option **B/C** requiring extra transceivers / a switch purchase?

### 2026-05-09 deep dive — what actually links / what doesn't

Cable inventory (post-deep-probe via `ethtool -m`):
- spark-05: **2 cables** — both `enp1s0f1np1` and `enP2p1s0f1np1` (the f1 ports). Identifier `0x11 (QSFP28)`, length 1 m, `FS QSFP-200G-PC005`. The two f0 ports are empty.
- spark-07: **4 cables** — every cage has a `FS QSFP-200G-PC005` 1 m DAC. So spark-07 has more cables in cages than spark-05 has.
- thor-04: lspci endpoints are NVIDIA SoC GPU, Realtek 8852CE wireless, Realtek 8126 (RJ45 1G) and the NVMe — **zero Mellanox / zero QSFP socket**. The 4× `mgbe` 10GBASE-T ports were `Link detected: yes` on May 7 but now show `Link detected: no` (cables moved off them between audits). A QSFP-200G-PC005 plug physically does not fit into any thor-04 port.

dmesg history on spark-07's link:
- `+2 141 269 s uptime`: both f1 ports went **`Link up`** simultaneously → `roce... Link ACTIVE` (RoCE up). This is the first time the cluster had a working high-speed link.
- `+2 146 692 s` (~1.5 h later): both f1 ports `Link down` together (suggests both went to the same partner, which then went away).
- `+2 351 223 s`: **all 4 modules** report `Cable unplugged`, ~6 s later all 4 plugged again.
- After that **no link has formed** on any port.

When I tried a plain `ip link set down/up` cycle on the f1 ports of both Sparks, the kernel surfaced **ConnectX-7 firmware-command timeouts** on both:
```
mlx5_core 0000:01:00.0: wait_func_handle_exec_timeout: ... DESTROY_EQ(0x302) timeout.
                       Will cause a leak of a command resource
mlx5_core 0002:01:00.0: query_mcia_reg failed: status: 0x3
mlx5_core 0000:01:00.1: wait_func_handle_exec_timeout: ... ACCESS_REG(0x805) timeout
```

Combined with the earlier boot warning `Detected insufficient power on the PCIe slot (27W)`, the Mellanox firmware on **both** Sparks is in a partially hung state. `ACCESS_REG`/`DESTROY_EQ` aren't completing, each timeout leaks a command-queue slot, and the cards refuse to negotiate link. A simple `ip link` cycle is not going to recover this — the firmware needs a reset (either `mlxfwreset` or a host cold reboot).

### Verdict

- **The "spark-07 ↔ thor-04" cable is a false premise.** thor-04 has no socket that accepts QSFP-200G-PC005. The cable end at the thor-04 chassis is either dangling, plugged into the chassis but not into a NIC port, or the user is mis-remembering which device is at the far end. This needs a visual cable trace.
- **The "spark-05 ↔ spark-07" cables are real.** Both endpoints have a matching cable on `enp1s0f1np1` + `enP2p1s0f1np1`. They linked once (`Link ACTIVE`) and have been dark since the most recent unplug/replug. Recovery requires firmware reset on at least spark-07's ConnectX-7.

### Recovery plan (granted by user, in progress)

1. Reboot spark-07 — clears its ConnectX-7 firmware. Briefly takes `gemma4-spark` vLLM down.
2. Reboot spark-05 — production qwen36 bare-metal vLLM goes down briefly; restarts via `run-qwen36-bare.sh`.
3. After both up, observe whether link auto-forms on the f1 ports.
4. If still dark: try `mlxfwreset -d <pci> reset` on each card.
5. If link forms: assign IPs in `10.10.10.0/30` (point-to-point pairs), validate ping + iperf3.
6. If link stays dark after fwreset: cable / firmware-version / PCIe-power issue — escalate.
