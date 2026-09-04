"""vidax -- a lightweight JAX/Flax inference engine and PyTorch-to-JAX weight
translator for modern Video Diffusion Transformers (DiTs), built for Google
Cloud TPUs.

Most users run the standalone scripts under ``examples/`` (see
``docs/models/``). When using ``vidax`` as a library, the building blocks are:

* ``vidax.core``        -- Pallas flash attention, 3D RoPE, TPU mesh helpers
* ``vidax.schedulers``  -- flow-matching / UniPC diffusion samplers
* ``vidax.translator``  -- PyTorch ``.safetensors``/``.pth`` -> Flax pytree
* ``vidax.models.<family>`` -- per-family DiT / VAE / text-encoder modules

See ``docs/library_usage.md`` for worked examples.
"""

__version__ = "0.1.0"

# Lazily re-export the handful of symbols that are useful straight off the
# top-level package, without importing every model subpackage (and its heavy
# deps) at ``import vidax`` time. See PEP 562.
_LAZY = {
    "load_torch_checkpoint_to_jax": "vidax.translator",
    "RectifiedFlowScheduler": "vidax.schedulers",
    "FlowUniPCMultistepScheduler": "vidax.schedulers",
    "build_tpu_mesh": "vidax.core",
    "dot_product_attention": "vidax.core",
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
