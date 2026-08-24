"""LTX-2.5's ancestral (SDE) Euler sampler for rectified-flow models, a
structural port of `EulerAncestralDiffusionStep`/`euler_ancestral_
denoising_loop` from `refs/LTX-2-main/packages/ltx-core/src/ltx_core/
components/diffusion_steps.py` and `refs/LTX-2-main/packages/ltx-pipelines/
src/ltx_pipelines/utils/samplers.py`.

Distinct from `vidax.schedulers.ltx_rectified_flow.RectifiedFlowScheduler`
(LTX-Video's, a plain deterministic Euler step): the **distilled**
checkpoint's own pipeline (`should_use_ancestral_sampler` returns `True`
for every 2.5-versioned checkpoint) uses `eta=1.0` -- each step advances
deterministically to an intermediate `sigma_down`, then re-noises back up
to `sigma_next`, rescaling by `alpha_next / alpha_down` to stay
variance-preserving. `eta=0` degenerates to a plain Euler step
(`sigma_down == sigma_next`, no noise) -- confirmed to be exactly what the
**`dev`/one-stage** pipeline actually uses by default
(`ltx_pipelines.utils.blocks.DiffusionStage.__call__`'s own defaults:
`loop=euler_denoising_loop`, `stepper=EulerDiffusionStep()` -- the plain,
non-ancestral step -- not the ancestral loop distilled's stage 1
explicitly opts into). One class covers both real recipes via `eta`.

Two sigma schedules are built in:

- `sampler="distilled"` (default): the distilled checkpoint's fixed
  9-value schedule (`DISTILLED_SIGMA_VALUES`, read directly from
  `ltx_pipelines.utils.constants` during this port), used with `eta=1.0`.
- `sampler="dev"`: the `dev`/one-stage pipeline's real schedule --
  `LTX2Scheduler.execute`'s token-count-dependent *shifted* sigma curve
  (`compute_shifted_sigmas`, ported from the same file), 30 steps, used
  with `eta=0.0` and real CFG (the reference's own `LTX_2_4_PARAMS`
  defaults for a `model_version >= (2, 4)` checkpoint, which a 2.5
  checkpoint resolves to since there's no `(2, 5)`-specific row:
  `num_inference_steps=30`, `video_cfg_guidance_scale=3.0`). STG and the
  audio/A2V/V2A guidance terms aren't ported (pure inference-loop
  refinements on a working base model, same reasoning as LTX-Video's own
  "plain CFG only" scope decision) -- plain CFG only, one constant
  `--guidance_scale` for the whole run rather than the reference's
  per-sigma-bucket guider-params schedule.

Not ported: the `res_2s` second-order sampler (two-stage-pipeline only,
out of scope -- see `docs/models/ltx2_5.md`'s Status section).

Noise for the ancestral re-injection is drawn from the caller's own JAX PRNG
key (`step`'s `noise` argument) rather than matching the reference's seeded
`torch.Generator` bit-for-bit -- mathematically equivalent, not literally
reproducible against the reference's own output for a given seed, same as
every other model in this repo (which all use JAX's own RNG already).
"""
import math
from typing import Optional

import jax.numpy as jnp

# `ltx_pipelines.utils.constants.DISTILLED_SIGMA_VALUES` -- read directly
# from the reference during this port, not derived. 8 steps (9 sigma
# boundaries including the terminal 0.0).
DISTILLED_SIGMA_VALUES = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)

# `ltx_pipelines.utils.constants.PipelineParams`/`LTX_2_3_PARAMS` -- the
# `dev` checkpoint's real recipe (a 2.5 checkpoint's `model_version`
# resolves to the newest defined generation at or below it, `(2, 4)`,
# since there is no `(2, 5)`-specific row in the reference's own
# `_PARAMS_SINCE_VERSION` table).
DEV_NUM_STEPS = 30
DEV_CFG_GUIDANCE_SCALE = 3.0

# `ltx_core.components.schedulers.LTX2Scheduler`'s own defaults.
_BASE_SHIFT_ANCHOR = 1024
_MAX_SHIFT_ANCHOR = 4096
_DEV_MAX_SHIFT = 2.05
_DEV_BASE_SHIFT = 0.95
_DEV_TERMINAL = 0.1


def compute_shifted_sigmas(
    steps: int, num_tokens: int,
    max_shift: float = _DEV_MAX_SHIFT, base_shift: float = _DEV_BASE_SHIFT,
    terminal: float = _DEV_TERMINAL, stretch: bool = True,
) -> jnp.ndarray:
    """`LTX2Scheduler.execute`: a token-count-dependent time-shift (the same
    family of formula as SD3/Flux's resolution-dependent shift --
    `exp(mu)/(exp(mu) + (1/t - 1))`, `mu` here linear in the *token count*
    rather than a fixed per-resolution constant) applied to a uniform
    `linspace(1, 0, steps+1)`, then optionally "stretched" so the last
    non-terminal sigma lands exactly at `terminal` -- both steps read
    directly from the reference, not assumed.

    Args:
        steps: number of sampling steps (returns `steps + 1` sigmas).
        num_tokens: `F' * H' * W'` -- the target latent's total token count
            (not including batch/channels). Larger videos shift the curve
            toward keeping more noise for more steps.

    Returns:
        `(steps + 1,)` descending sigmas, `sigmas[0] == 1.0`,
        `sigmas[-1] == 0.0`.
    """
    sigmas = jnp.linspace(1.0, 0.0, steps + 1)

    mm = (max_shift - base_shift) / (_MAX_SHIFT_ANCHOR - _BASE_SHIFT_ANCHOR)
    b = base_shift - mm * _BASE_SHIFT_ANCHOR
    sigma_shift = num_tokens * mm + b

    shifted = jnp.where(
        sigmas != 0,
        math.exp(sigma_shift) / (math.exp(sigma_shift) + (1.0 / jnp.where(sigmas != 0, sigmas, 1.0) - 1.0)),
        0.0,
    )

    if stretch:
        # Rescale (1 - sigma) so the last non-zero sigma maps exactly to
        # `terminal`, keeping sigmas[0]=1 and the trailing 0.0 fixed.
        last_nonzero = shifted[-2]
        scale_factor = (1.0 - last_nonzero) / (1.0 - terminal)
        stretched = 1.0 - (1.0 - shifted) / scale_factor
        shifted = jnp.where(sigmas != 0, stretched, 0.0)

    return shifted.astype(jnp.float32)


class AncestralEulerScheduler:
    """Ancestral Euler sampler. `sigmas`/`timesteps` are the same quantity
    (matching `vidax.schedulers.ltx_rectified_flow.RectifiedFlowScheduler`'s
    own convention) -- the DiT's own `timestep_scale_multiplier` handles the
    training-scale rescale internally (see `vidax.models.ltx2_5.dit`).
    """

    def __init__(
        self, sampler: str = "distilled", sigmas: Optional[jnp.ndarray] = None,
        eta: Optional[float] = None, s_noise: float = 1.0,
        num_steps: int = DEV_NUM_STEPS, num_tokens: Optional[int] = None,
    ):
        """
        Args:
            sampler: `"distilled"` (default, `eta=1.0` unless overridden) or
                `"dev"` (`eta=0.0` unless overridden, needs `num_tokens`).
                Ignored when `sigmas=` is given explicitly.
            sigmas: explicit sigma schedule override.
            eta: defaults to the real recipe's own value per `sampler`
                (`1.0` distilled, `0.0` dev) -- see module docstring.
            num_steps: `sampler="dev"` only -- step count for
                `compute_shifted_sigmas` (real recipe: `30`).
            num_tokens: `sampler="dev"` only -- the target latent's
                `F' * H' * W'` token count, required for
                `compute_shifted_sigmas`.
        """
        if sigmas is not None:
            self.sigmas = jnp.asarray(sigmas, dtype=jnp.float32)
            default_eta = 1.0
        elif sampler == "distilled":
            self.sigmas = jnp.asarray(DISTILLED_SIGMA_VALUES, dtype=jnp.float32)
            default_eta = 1.0
        elif sampler == "dev":
            if num_tokens is None:
                raise ValueError("sampler='dev' requires `num_tokens` (the target latent's F'*H'*W').")
            self.sigmas = compute_shifted_sigmas(num_steps, num_tokens)
            default_eta = 0.0
        else:
            raise NotImplementedError(
                f"No built-in sigma schedule for sampler={sampler!r} -- pass `sigmas=` explicitly.")
        self.num_steps = self.sigmas.shape[0] - 1
        self.eta = default_eta if eta is None else eta
        self.s_noise = s_noise

    def step(
        self,
        denoised_sample: jnp.ndarray,
        sample: jnp.ndarray,
        step_index: int,
        noise: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """One ancestral Euler step: `x_t -> x_{t-1}`.

        Args:
            denoised_sample: `x_0` estimate (`sample - velocity * sigma`,
                see `vidax.models.ltx2_5.dit.LTXDiT`'s velocity-prediction
                convention -- the caller computes this, not this method).
            sample: current noisy latent `x_t`, `(B, N, C)` (or any shape;
                elementwise).
            step_index: current step index into `self.sigmas`.
            noise: required when `self.eta > 0` (the default) -- standard
                normal, same shape as `sample`.

        Returns:
            `x_{t-1}`, or `denoised_sample` directly when `sigma_next == 0`
            (the terminal step).
        """
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]

        x = sample.astype(jnp.float32)
        denoised = denoised_sample.astype(jnp.float32)

        downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * self.eta
        sigma_down = sigma_next * downstep_ratio
        sigma_down_ratio = sigma_down / sigma
        x_next = sigma_down_ratio * x + (1.0 - sigma_down_ratio) * denoised

        if self.eta > 0:
            if noise is None:
                raise ValueError("AncestralEulerScheduler.step requires `noise` when eta > 0.")
            alpha_next = 1.0 - sigma_next
            alpha_down = 1.0 - sigma_down
            renoise_coeff = jnp.sqrt(jnp.clip(
                sigma_next ** 2 - sigma_down ** 2 * alpha_next ** 2 / alpha_down ** 2, min=0.0))
            x_next = (alpha_next / alpha_down) * x_next + noise.astype(jnp.float32) * self.s_noise * renoise_coeff

        x_next = jnp.where(sigma_next == 0, denoised, x_next)
        return x_next.astype(sample.dtype)
