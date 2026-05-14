# Troubleshooting

## `Set REMOTE=user@host`

Export `REMOTE` before using `deploy-model.sh` or the wrapper scripts.

```bash
export REMOTE=jetbrains@your-spark-host
```

## First Start Downloads, Later Starts Do Not

`run-qwen3.sh` and `run-nemotron3.sh` are cache-aware:

- if the model snapshot already exists in `~/.cache/huggingface`, they stay offline
- if the snapshot is missing, they allow a first-run download

`run-gpt-oss.sh` is only partially offline-first: if the pinned snapshot is missing or incomplete it will repair the
cache, and a cold Docker image build still needs network access unless `gpt-oss-custom:latest` or its build layers
are already present.

## SSH host key failures

The scripts use `StrictHostKeyChecking=accept-new`, which accepts unknown keys but does not fix changed keys.
If the Spark was reimaged or its IP was reused, remove the stale key from `known_hosts` and retry.

## CRLF shell-script failures

Symptoms:

- `\r: command not found`
- `syntax error: unexpected end of file`

Fix:

```bash
sed -i 's/\r$//' *.sh
```

## Slow first downloads

Without `HF_TOKEN`, initial model downloads are much slower and more likely to hit rate limits.

## Old services respawn containers

If `qwen3-vllm`, `qwen3-coder-server`, or `gpt-oss-server` is still enabled, it can restart a container behind the
deploy script. Disable the unit with `sudo systemctl disable --now ...` on the Spark.

## GPT-OSS takes a long time to start

The GPT-OSS path has two expensive phases:

1. building the custom image
2. loading the model checkpoint

Check:

```bash
docker logs -f gpt-oss
```

## Smoke tests time out on the first request

The load scripts default to `REQUEST_TIMEOUT=120`. Increase it if you want a larger cold-start margin:

```bash
REQUEST_TIMEOUT=240 bash qwen3-load.sh localhost:8000
```

## `spark-NN.local` mDNS resolution intermittently fails

Symptom: `ssh: Could not resolve hostname spark-05.local: nodename nor servname provided, or not known`,
or new SSH connections to a healthy Spark time out while an existing session is fine.

Common when the Spark is under heavy I/O (large model downloads, container churn). The fix is to route
through Tailscale instead — every Spark in `~/.ssh/config` should list its `*.tail59a662.ts.net` name as
`HostName`:

```
Host spark-05 spark-05.local spark-05.labs.intellij.net spark-05.tail59a662.ts.net
    HostName spark-05.tail59a662.ts.net
    User jetbrains
    IdentityFile ~/.ssh/spark05
    IdentitiesOnly yes
```

## Deploy hangs at "Waiting for ... health"

The deploy uses one long-lived SSH that runs a `curl` poll loop on the Spark. If that SSH is killed mid-run
(network glitch, sshd restart), the Mac side keeps the half-dead TCP open and waits forever even though
the model is already serving.

Verify the model directly via the Tailscale IP, then `kill` the local ssh PID:

```bash
curl -sf http://<tailscale-ip>:8005/health     # 200 means it is up
lsof -nP -iTCP:22 -sTCP:ESTABLISHED | grep ssh # find the half-dead PID
kill <pid>
```

## Permission denied writing `~/.cache/huggingface`

Symptom: `llama-benchy` or other host-side tools fail with `PermissionError` under `~/.cache/huggingface/hub/models--*`.
The vLLM container runs as root and creates cache directories owned by root on the host bind mount.
Reset ownership when host-side tooling needs to write:

```bash
sudo chown -R "$USER":"$USER" ~/.cache/huggingface ~/.cache/vllm
```

## Qwen3.6 MTP — `cudaErrorIllegalAddress` / `EngineDeadError`

Symptoms: `vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.` followed by
`torch.AcceleratorError: CUDA error: an illegal memory access was encountered`. Reproducible with
`QWEN36_NUM_SPECULATIVE_TOKENS=2` at `concurrency=8`, and even at `concurrency=1` with `=3`,
on the pinned `scitrera/dgx-spark-vllm:0.17.0-t5` image.

Workaround: keep `QWEN36_NUM_SPECULATIVE_TOKENS` at `0` or `1`. Re-evaluate after the image is
upgraded — this looks like a bug in vLLM's MTP path on this build, not a Qwen3.6 limitation.

Once the engine has crashed, `/health` may still return 200 but every completion request returns HTTP 500.
Recovery: `docker rm -f vllm_qwen36 && cd ~/spark-setup && QWEN36_NUM_SPECULATIVE_TOKENS=1 bash run-qwen36.sh`.

## Bare-metal vLLM crashes on first request: `FileNotFoundError: 'ninja'`

FlashInfer JIT-compiles attention kernels on first inference. It needs the system `ninja` build
tool, not the Python `ninja` package alone (subprocess from FlashInfer doesn't always inherit the
venv `bin/`).

Install once on the Spark:

```bash
sudo apt-get install -y python3.12-dev build-essential ninja-build
```

The `python3.12-dev` and `build-essential` are also needed because `fastsafetensors` (a vLLM dep)
ships only an aarch64 sdist and builds from source.

## Orphan `EngineCore` holds GPU memory after vLLM restart

After killing a bare-metal `vllm serve`, the child `VLLM::EngineCore` process can survive and keep
GPU memory pinned (~50–100 GB on Spark). `pgrep -f vllm` does **not** match it (its command name
is the abbreviated `VLLM::EngineCor`). Symptoms on the next launch:
`ValueError: Free memory on device cuda:0 (12.1/121.69 GiB) on startup is less than desired GPU memory utilization`.

Clean reset that catches everything holding the GPU:

```bash
for p in $(sudo fuser /dev/nvidia0 2>/dev/null | tr ' ' '\n'); do
  [ -n "$p" ] && [ "$p" != "1458" ] && sudo kill -9 $p 2>/dev/null
done
sleep 5
```

The PID `1458` exception is for `nvidia-persistenced`, which holds `/dev/nvidia0` as root and should
not be killed. Verify with `sudo fuser -v /dev/nvidia0`.

## Hugging Face gated repo (`401 Cannot access gated repo`)

Symptom on vLLM serve startup:
`OSError: You are trying to access a gated repo. ... 401 Client Error.` plus
`Access to model <X> is restricted. You must have access to it and be authenticated to access it. Please log in.`

Two parts to get right:

1. **The HF account itself must be granted access.** Visit the model page on huggingface.co and click "Request access". You'll receive an email when granted.
2. **`HF_TOKEN` must be exported in the launching shell** (vLLM picks it up via env, not just from `~/.cache/huggingface/token`). The token at `~/.cache/huggingface/token` may also be expired — the canonical place we use is `~/.hf`. Verify with:
   ```bash
   HF_TOKEN=$(cat ~/.hf) curl -s -H "Authorization: Bearer $HF_TOKEN" \
     https://huggingface.co/api/whoami-v2 | python3 -m json.tool
   ```
   should return your account JSON. If it returns `{"error":"Invalid username or password."}` the token itself is invalid — generate a fresh one at <https://huggingface.co/settings/tokens> with **read** scope.
3. Mirror the working token to the Spark before launching vLLM with a gated draft model:
   ```bash
   ssh spark-05 "echo -n '$(cat ~/.hf)' > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token"
   ```

## Field Diagnostics (`partnerdiag --field`) won't start

Two preconditions tend to block it:

1. **Secure Boot must be disabled** in UEFI. Verify on the Spark:
   ```bash
   mokutil --sb-state          # must print: SecureBoot disabled
   ```
   The setting can only be changed in firmware setup (Esc/Del during POST). After the diag completes you can leave Secure Boot off or re-enable it.
2. **`nvidia_drm` must be unloadable.** On a default DGX OS install, `gdm` (GNOME Display Manager) holds it via Xorg/gnome-shell/mutter. The diag loops `rmmod: ERROR: Module nvidia_drm is in use` forever and never starts the actual tests. Stop `gdm` first:
   ```bash
   sudo systemctl stop gdm
   sudo fuser -v /dev/nvidia0           # should show nothing but nvidia-persistenced
   # ... run partnerdiag --field ...
   sudo systemctl start gdm             # restore after the diag finishes
   ```

The diag itself is at `/opt/nvidia/dgx-spark-fieldiag/partnerdiag`, takes ~30 min, and reports PASS/FAIL. Logs land in the directory passed via `--log <dir>`.

## ConnectX-7 NICs vanish from `lspci` post-boot (chronic PCIe downtrain)

This is a known DGX Spark hardware defect — see <docs/spark-05-pcie-defect.md> for the full case study and `docs/spark-05-pcie-defect-bundle/` for the evidence package used in our RMA submission.

Quick diagnostic on any Spark:

```bash
ssh <spark-host> 'sudo lspci -vvv -s 0000:00:00.0 | grep -E "LnkCap:|LnkSta:"; \
                  sudo lspci -vvv -s 0002:00:00.0 | grep -E "LnkCap:|LnkSta:"'
```

A healthy Spark reports `LnkSta: Speed 32GT/s, Width x4` on both bridges. If you see
`LnkSta: Speed 2.5GT/s, Width x4` instead, the ConnectX-7 PCIe link has downtrained to PCIe 1.0 and the cards will quietly drop off the bus after `mlx5_core E-Switch: cleanup`. **No software fix exists** — NVIDIA support directs reporters to Field Diagnostics → RMA.

While the unit is in this state:

- Do **not** run `echo 1 | sudo tee /sys/bus/pci/rescan` — that wedges the kernel.
- Do **not** `ip link set <mlx5-port> down/up` — leaks `mlx5_core` command-queue resources and the next `sudo reboot` won't bring the host back.
- Recovery from any of those wedges requires a **hard power-cycle with QSFP cages empty**.

## llama-benchy tries to download `gpt2`

llama-benchy defaults to `gpt2` for token counting. On a fresh Spark this fails (no `HF_TOKEN` for the
hub fetch and the cache may be root-owned). Pass the served model's tokenizer explicitly:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
llama-benchy --tokenizer Qwen/Qwen3.6-27B-FP8 --model Qwen/Qwen3.6-27B-FP8 ...
```

`bench-qwen36.sh` already does this.
