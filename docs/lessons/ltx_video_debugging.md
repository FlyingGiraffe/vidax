# LTX-Video Debugging Lessons

LTX-Video's port was verified via direct numerical comparison against the
actual reference PyTorch implementation, not just real-checkpoint
end-to-end runs. See [`docs/models/ltx_video.md`](../models/ltx_video.md)
for the full port and its current status.

## Verification methodology: a throwaway conda env for bit-exact checks

Worth keeping in mind for any cross-framework port, not just this one: at
JAX's *default* matmul precision, this port's outputs only agreed with the
reference to ~2 decimal places (correlation ~0.9999, max diff ~0.02) even
though both were numerically correct — JAX defaults to a lower-precision
matmul algorithm on this backend for speed. Setting
`jax.config.update("jax_default_matmul_precision", "highest")` before
comparing collapsed the gap to `~3e-5` max diff (correlation
`0.999999999984`). A cross-framework numerical comparison that doesn't
force this can look like a real correctness bug when it isn't one.
