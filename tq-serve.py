#!/usr/bin/env python3
"""
TurboQuant + vLLM serve wrapper.

Runs turboquant.vllm_attn_backend.enable_no_alloc() BEFORE vLLM CLI starts so
the Executor + GPUModelRunner monkey-patches are in place when the engine boots.
Then hands off to `vllm serve <args>` exactly as the regular CLI would.

REQUIRED env (set in the shell, NOT inside this script):
    VLLM_ENABLE_V1_MULTIPROCESSING=0

Optional env:
    TQ_KEY_BITS         (default 3)
    TQ_VALUE_BITS       (default 2)
    TQ_BUFFER_SIZE      (default 128)
    TQ_INITIAL_LAYERS   (default 4)

Usage (run in the bare-metal venv):

    VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
    python tq-serve.py serve Qwen/Qwen3.6-27B-FP8 --port 8006 \\
      --max-model-len 262144 --gpu-memory-utilization 0.80 \\
      --reasoning-parser qwen3 --enable-prefix-caching ...
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

KEY_BITS = int(os.environ.get("TQ_KEY_BITS", "3"))
VALUE_BITS = int(os.environ.get("TQ_VALUE_BITS", "2"))
BUFFER_SIZE = int(os.environ.get("TQ_BUFFER_SIZE", "128"))
INITIAL_LAYERS = int(os.environ.get("TQ_INITIAL_LAYERS", "4"))

# Module-level so worker subprocesses (if any spawn) re-apply the patch on import.
from turboquant.vllm_attn_backend import enable_no_alloc

enable_no_alloc(
    key_bits=KEY_BITS,
    value_bits=VALUE_BITS,
    buffer_size=BUFFER_SIZE,
    initial_layers_count=INITIAL_LAYERS,
)


def _main():
    import sys
    print(
        f"[tq-serve] enabled: key_bits={KEY_BITS} value_bits={VALUE_BITS} "
        f"buffer_size={BUFFER_SIZE} initial_layers={INITIAL_LAYERS}",
        flush=True,
    )
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") != "0":
        print(
            "[tq-serve] WARNING: VLLM_ENABLE_V1_MULTIPROCESSING is not 0 — "
            "monkey-patches may not propagate to spawned workers.",
            flush=True,
        )
    from vllm.entrypoints.cli.main import main as vllm_main
    sys.argv = ["vllm"] + sys.argv[1:]
    sys.exit(vllm_main())


if __name__ == "__main__":
    _main()
