"""Patch vLLM weight loader to skip unknown checkpoint weights instead of crashing.

vLLM 0.17.x _load_module raises ValueError for checkpoint weights that don't
match any model parameter. Gemma 4 checkpoints include layer_scalar which
transformers doesn't expose as nn.Parameter yet. This patch makes it a warning.
"""
import pathlib

p = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages"
    "/vllm/model_executor/models/utils.py"
)
src = p.read_text()

old = """\
                raise ValueError(msg)

    @support_quantized_model_reload_from_hp_weights
    def load_weights("""

new = """\
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "Skipping unknown weight: %s", prefix)
                continue

    @support_quantized_model_reload_from_hp_weights
    def load_weights("""

assert old in src, f"Patch target not found in {p}"
p.write_text(src.replace(old, new, 1))
print("Patched _load_module to skip unknown weights")
