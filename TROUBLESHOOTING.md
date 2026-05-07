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

## llama-benchy tries to download `gpt2`

llama-benchy defaults to `gpt2` for token counting. On a fresh Spark this fails (no `HF_TOKEN` for the
hub fetch and the cache may be root-owned). Pass the served model's tokenizer explicitly:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
llama-benchy --tokenizer Qwen/Qwen3.6-27B-FP8 --model Qwen/Qwen3.6-27B-FP8 ...
```

`bench-qwen36.sh` already does this.
