"""LTX-Video's Rectified Flow scheduler (Flax/JAX-friendly, no PyTorch
dependency), a structural port of the reference `RectifiedFlowScheduler`
(`refs/LTX-Video-main/ltx_video/schedulers/rf.py`).

A deliberately separate file from `vidax.schedulers.flow_match.
RectifiedFlowScheduler` (Wan's), not an extension of it -- two differences
that a shared implementation could not paper over without touching Wan's
scheduler:

- **Per-token timestep support**, required for I2V's conditioning-mask-
  driven per-token denoising (see `examples/generate_ltx_video.py`): `step`
  accepts `timestep` shaped `(B,)` (T2V, one timestep per sample) *or*
  `(B, N)` (I2V, one timestep per token). Wan's scheduler only ever handles
  the former.
- **`LinearQuadratic`/`Constant` sampler schedules** (`sampler=` in the
  checkpoint's own embedded scheduler config -- every released LTX-Video
  0.9.8 checkpoint uses `"LinearQuadratic"`), on top of the plain
  `"Uniform"` linspace Wan's own `shift`-only schedule effectively is.

Not ported for this first version (raises if requested, rather than
guessing at an unverified implementation): `shifting="SD3"`/`"SimpleDiffusion"`
resolution-dependent timestep shifting, and `stochastic_sampling=True`'s
re-noising step -- neither is used by either downloaded checkpoint
(`shifting: null`, `stochastic_sampling: false` in both configs' embedded
metadata).
"""
from typing import Optional

import jax.numpy as jnp


def _linear_quadratic_schedule(num_steps: int, threshold_noise: float = 0.025, linear_steps: Optional[int] = None):
    """Reference `linear_quadratic_schedule`: a linear ramp for the first
    `linear_steps`, then a quadratic tail, expressed in `sigma` space
    (1.0 = pure noise, descending to ~0.0) -- `num_steps` sigmas, the
    reference's own trailing `[:-1]` slice of a `num_steps + 1`-long
    schedule already applied.
    """
    if num_steps == 1:
        return jnp.array([1.0])
    if linear_steps is None:
        linear_steps = num_steps // 2

    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps ** 2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
    const = quadratic_coef * (linear_steps ** 2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i ** 2) + linear_coef * i + const for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return jnp.array(sigma_schedule[:-1], dtype=jnp.float32)


def _time_shift(mu: float, sigma: float, t: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(mu) / (jnp.exp(mu) + (1 / t - 1) ** sigma)


class RectifiedFlowScheduler:
    """Euler sampler for LTX-Video's Rectified Flow / Flow Matching model.

    `sigmas`/`timesteps` are the same quantity here (unlike Wan's scheduler,
    the reference never separately scales by `num_train_timesteps` at the
    scheduler level -- that scaling happens inside the DiT itself, via
    `timestep_scale_multiplier`; see `vidax.models.ltx_video.dit`).
    """

    def __init__(self, num_steps: int = 30, sampler: str = "LinearQuadratic", shift: Optional[float] = None):
        self.num_steps = num_steps
        self.sampler = sampler
        if sampler == "Uniform":
            sigmas = jnp.linspace(1.0, 1.0 / num_steps, num_steps)
        elif sampler == "LinearQuadratic":
            sigmas = _linear_quadratic_schedule(num_steps)
        elif sampler == "Constant":
            if shift is None:
                raise ValueError("`shift` must be provided for the 'Constant' sampler.")
            sigmas = _time_shift(shift, 1.0, jnp.linspace(1.0, 1.0 / num_steps, num_steps))
        else:
            raise ValueError(f"Unsupported sampler: {sampler!r}")
        self.sigmas = sigmas
        self.timesteps = sigmas

    def step(
        self,
        model_output: jnp.ndarray,
        timestep: jnp.ndarray,
        sample: jnp.ndarray,
    ) -> jnp.ndarray:
        """Euler step: finds each element's next-lower scheduled sigma and
        steps to it (`sample - dt * model_output`), matching the
        reference's "not required to be exactly a scheduled timestep"
        behavior -- needed since I2V's per-token timestep is clamped
        per-element and generally isn't itself one of `self.sigmas`.

        Args:
            model_output: (B, N, C) predicted velocity.
            timestep: (B,) (T2V) or (B, N) (I2V) current timestep(s), in
                `[0, 1]` sigma space.
            sample: (B, N, C) current noisy latent tokens.

        Returns:
            (B, N, C) the stepped sample.
        """
        t_eps = 1e-6
        timesteps_padded = jnp.concatenate([self.sigmas, jnp.zeros((1,), dtype=self.sigmas.dtype)])

        # Broadcast `timesteps_padded` (T+1,) against `timestep`'s own rank
        # (1 for T2V's (B,), 2 for I2V's (B, N)) by appending trailing axes.
        t_pad = timesteps_padded.reshape((-1,) + (1,) * timestep.ndim)
        lower_mask = t_pad < (timestep[None] - t_eps)
        lower_timestep = jnp.max(jnp.where(lower_mask, t_pad, 0.0), axis=0)
        dt = timestep - lower_timestep
        # Append trailing axes to broadcast against `sample`'s channel dim
        # (and, for T2V's (B,) timestep, its token dim too).
        dt = dt.reshape(dt.shape + (1,) * (sample.ndim - dt.ndim))

        return sample - dt * model_output

    def add_noise(self, original_samples: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray) -> jnp.ndarray:
        """Forward flow-matching process: `alphas * x0 + sigmas * noise`,
        `sigmas == timesteps`. Used both to build the initial fully-noised
        latent-adjacent quantities and (I2V) to re-noise conditioning
        latents (`image_cond_noise_scale`) -- see
        `examples/generate_ltx_video.py`.
        """
        sigmas = timesteps.reshape(timesteps.shape + (1,) * (original_samples.ndim - timesteps.ndim))
        alphas = 1 - sigmas
        return alphas * original_samples + sigmas * noise
