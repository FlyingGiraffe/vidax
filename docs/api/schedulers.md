# `vidax.schedulers` — diffusion samplers

One sampler per model's native schedule, all pure-JAX (no PyTorch), all
`jax.jit`-friendly. The two general-purpose ones are re-exported from the
package top level:

```python
from vidax.schedulers import RectifiedFlowScheduler, FlowUniPCMultistepScheduler, UniPCState
```

The family-specific ones are imported from their submodule
(`from vidax.schedulers.cogvideox import CogVideoXDPMScheduler`).

**Calling convention** shared by all of them: build the scheduler once, then
run a plain Python loop that (1) calls your DiT to get a model output for the
current step, (2) calls `scheduler.step(...)` to advance the latent. The
scheduler owns no loop counter — you pass `step_index` (or `timestep`)
explicitly. For the flow-matching samplers, feed the DiT's timestep
embedding **`scheduler.timesteps[i]`** (on the training `~[0, 1000]` scale),
not the raw sigma.

---

## `RectifiedFlowScheduler` (Wan)

`vidax.schedulers.flow_match.RectifiedFlowScheduler` — deterministic Euler
sampler for rectified-flow / flow-matching models, matching Wan2.1/2.2's
schedule (minus its higher-order UniPC integration).

```python
RectifiedFlowScheduler(
    num_steps: int = 50,
    num_train_timesteps: int = 1000,
    shift: float = 5.0,          # resolution-dependent noise-schedule warp; 5.0 = Wan T2V default
)
```

| Attribute | Shape | Meaning |
| --- | --- | --- |
| `.sigmas` | `(num_steps + 1,)` | Flow coefficient, `1.0 → 0.0`, `shift`-warped. `x_new = x − v·Δsigma` steps in this space. |
| `.timesteps` | `(num_steps + 1,)` | `sigmas * num_train_timesteps` — the value to feed the DiT's timestep embedding. |
| `.num_steps` | — | `int` |

```python
step(model_output: jax.Array, step_index: int, x: jax.Array) -> jax.Array
```

One Euler step from `step_index` to `step_index + 1`. `model_output` is the
DiT's predicted velocity `v_t`; `x` the current latent. Returns the next
latent.

```python
sched = RectifiedFlowScheduler(num_steps=50, shift=5.0)
x = initial_noise
for i in range(sched.num_steps):
    v = dit.apply(params, x, sched.timesteps[i], ...)
    x = sched.step(v, i, x)
```

---

## `FlowUniPCMultistepScheduler`

`vidax.schedulers.unipc.FlowUniPCMultistepScheduler` — training-free UniPC
predictor-corrector ODE solver (flow-matching adaptation of
[arXiv:2302.04867](https://arxiv.org/abs/2302.04867)). Fits a local
polynomial through several past `(x0-prediction, sigma)` pairs, so it
reaches Euler-quality samples in far fewer steps (Cosmos uses
`solver_order=2`, `num_steps=35`).

```python
FlowUniPCMultistepScheduler(
    num_steps: int = 35,
    num_train_timesteps: int = 1000,
    shift: float = 5.0,
    solver_order: int = 2,
    solver_type: str = "bh2",              # only "bh2" is ported (raises otherwise)
    lower_order_final: bool = True,        # drop to lower order over the last few steps
    disable_corrector: Sequence[int] = (), # step indices at which to skip the corrector
    use_karras_sigmas: bool = False,       # Cosmos3-Nano's schedule: Karras ramp + flow remap
    karras_sigma_min: float = 0.147,
    karras_sigma_max: float = 200.0,
    karras_rho: float = 7.0,
)
```

Attributes: `.sigmas` `(num_steps + 1,)` (starts at `1 − 1/num_train_timesteps`,
**not** exactly 1.0 — load-bearing for UniPC's `log(1 − sigma)`),
`.timesteps`, `.num_steps`, `.solver_order`.

```python
init_state() -> UniPCState
step(state: UniPCState,
     model_output: jax.Array,
     step_index: int,
     x: jax.Array) -> tuple[UniPCState, jax.Array]
```

`step` runs one predictor-corrector step and returns `(new_state, new_x)`.
Thread the state through the loop:

```python
sched = FlowUniPCMultistepScheduler(num_steps=35, solver_order=2, shift=5.0)
state = sched.init_state()
x = initial_noise
for i in range(sched.num_steps):
    v = dit.apply(params, x, sched.timesteps[i], ...)
    state, x = sched.step(state, v, i, x)
```

### `UniPCState`

Frozen dataclass, registered as a JAX pytree (so it passes through `jax.jit`
/ `lax.scan`). Fields — you never build these by hand, just pass the object
through:

| Field | Kind | Meaning |
| --- | --- | --- |
| `model_outputs` | array data (`None` until first step) | `(solver_order, *x.shape)` rolling history of converted (x0-prediction) outputs, newest last. |
| `last_sample` | array data (`None` until first step) | Sample before the most recent predictor step; `None` disables the corrector on step 0. |
| `this_order` | static meta field | Order to use in the next corrector call (warmup ramp `1, 2, …, solver_order`). Static → at most `solver_order` retraces per run. |

---

## Family-specific samplers

### CogVideoX — `vidax.schedulers.cogvideox`

Structural ports of diffusers' `CogVideoXDDIMScheduler` /
`CogVideoXDPMScheduler` (v-prediction, `scaled_linear` betas, `trailing`
spacing, zero-terminal-SNR). Shared constructor:

```python
CogVideoXDDIMScheduler(num_inference_steps, *,
    num_train_timesteps=1000, beta_start=0.00085, beta_end=0.0120,
    snr_shift_scale=1.0, set_alpha_to_one=True, rescale_betas_zero_snr=True)
CogVideoXDPMScheduler(...same...)
```

`snr_shift_scale` is `3.0` for CogVideoX-2b, `1.0` otherwise (see
`vidax.models.cogvideo.configs`). Attributes: `.timesteps`
`(num_inference_steps,)`, `.init_noise_sigma = 1.0`.

```python
# DDIM (deterministic):
CogVideoXDDIMScheduler.step(model_output, timestep: int, sample)
    -> (prev_sample, pred_original_sample)

# DPM-Solver++ multistep:
CogVideoXDPMScheduler.step(model_output, old_pred_original_sample,
    timestep: int, timestep_back, sample, noise, noise2=None)
    -> (prev_sample, pred_original_sample)
```

For DPM: `old_pred_original_sample` is the previous step's returned
`pred_original_sample` (`None` on step 0); `timestep_back` the previous
step's timestep (`None` on step 0); `noise`/`noise2` are caller-drawn
Gaussians shaped like `sample`. The `use_dynamic_cfg` guidance schedule
lives in `examples/generate_cogvideox.py`, not the scheduler.

### LTX-Video — `vidax.schedulers.ltx_rectified_flow`

> **Name collision:** this module also defines a class named
> `RectifiedFlowScheduler`, distinct from the Wan one re-exported at
> `vidax.schedulers.RectifiedFlowScheduler`. Import it by its full path.

```python
RectifiedFlowScheduler(num_steps: int = 30,
                       sampler: str = "LinearQuadratic",   # or "Uniform" | "Constant"
                       shift: float | None = None)         # required for "Constant"
```

Here `.sigmas` and `.timesteps` are the same array (the training-scale
rescale happens inside the DiT via `timestep_scale_multiplier`).

```python
step(model_output, timestep, sample) -> sample
```

`timestep` may be `(B,)` (T2V, one per sample) **or** `(B, N)` (I2V, one per
token); `model_output` and `sample` are `(B, N, C)`. Steps each element to
its next-lower scheduled sigma.

```python
add_noise(original_samples, noise, timesteps) -> noisy
```

Forward flow-matching process `(1 − sigma)·x0 + sigma·noise` — used to build
the initial noised latent and (I2V) to re-noise conditioning latents.

### LTX-2.5 — `vidax.schedulers.ltx2_5_ancestral_euler`

Ancestral (SDE) Euler sampler. `eta=1.0` (distilled recipe) advances to an
intermediate `sigma_down` then re-noises up to `sigma_next`; `eta=0` (dev
recipe) degenerates to plain Euler.

```python
AncestralEulerScheduler(
    sampler: str = "distilled",          # "distilled" (eta=1.0) | "dev" (eta=0.0)
    sigmas: jax.Array | None = None,      # explicit schedule override
    eta: float | None = None,            # overrides the per-sampler default
    s_noise: float = 1.0,
    num_steps: int = 30,                  # sampler="dev" only
    num_tokens: int | None = None,        # sampler="dev" only; default 4096 (resolution-independent)
)
```

`"distilled"` uses a fixed 8-step sigma table; `"dev"` builds a
token-count-shifted schedule via `compute_shifted_sigmas`. Attributes:
`.sigmas`, `.num_steps` (`len(sigmas) - 1`), `.eta`.

```python
step(denoised_sample, sample, step_index: int, noise=None) -> x_next
```

`denoised_sample` is the `x0` estimate (`sample − velocity·sigma`; the
caller computes it). `noise` (standard normal, shape of `sample`) is
**required when `eta > 0`**. Returns `x_{t-1}`, or `denoised_sample`
directly at the terminal step.

```python
compute_shifted_sigmas(steps: int, num_tokens: int,
    max_shift=2.05, base_shift=0.95, terminal=0.1, stretch=True) -> jax.Array
```

Module-level helper: a token-count-dependent time-shift (SD3/Flux-family
formula, `mu` linear in token count) applied to `linspace(1, 0, steps+1)`,
optionally stretched so the last non-terminal sigma lands at `terminal`.
Returns `(steps + 1,)` descending sigmas, `[0] == 1.0`, `[-1] == 0.0`.
