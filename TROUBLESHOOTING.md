# Troubleshooting

## `Set REMOTE=user@host`

Export `REMOTE` before using `deploy-model.sh` or the wrapper scripts.

```bash
export REMOTE=jetbrains@your-spark-host
```

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
