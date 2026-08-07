"""Flow-matching UniPC multistep scheduler.

Port of `FlowUniPCMultistepScheduler`
(refs/cosmos-predict2.5-main/cosmos_predict2/_src/predict2/models/fm_solvers_unipc.py),
itself a flow-matching adaptation of diffusers'
`UniPCMultistepScheduler` (https://arxiv.org/abs/2302.04867). UniPC is a
training-free predictor-corrector ODE solver: each step first *predicts*
`x_{t-1}` from a short history of previous model outputs using a
higher-order (up to solver_order) multistep extrapolation, then *corrects*
the current sample using the freshly computed model output before that
history is used for the next prediction. Because it fits a local polynomial
through several past (x0-prediction, sigma) pairs rather than taking a
single first-order Euler step (`RectifiedFlowScheduler` in `flow_match.py`),
it reaches the same sample quality in substantially fewer steps -- Cosmos2.5
uses it at inference with `solver_order=2`, `num_steps=35`, `shift=5.0`,
versus the ~50 Euler steps Wan2.1/2.2 use.

Sigma/timestep schedule construction (linear sigma grid warped by `shift`)
follows the same style as `RectifiedFlowScheduler` in `flow_match.py`, with
one deliberate, load-bearing difference at the top end: the grid starts at
`sigma_max = 1 - 1 / num_train_timesteps` (matching the reference's
`self.sigma_max`, derived from its 1000-step training grid) rather than
`flow_match.py`'s exact `1.0`. Euler stepping never takes `log(alpha_t) =
log(1 - sigma_t)`, so `sigma = 1.0` (alpha = 0) is harmless there; UniPC's
predictor/corrector *do* take that log when building `lambda_si` for
history entries, and solver_order >= 2 kicks in as early as `step_index=1`
(see the `lower_order_nums` warmup below) -- at which point it looks back
at `sigmas[0]`. `log(1 - 1.0) = log(0) = -inf` poisons the linear systems
(`R`/`b`) solved for the corrector's coefficients at order 2 and up,
producing NaNs from the very first corrected step; this was caught by
running the port (see the accompanying sanity check). Starting at `1 -
1/num_train_timesteps` instead (as the reference does) keeps `log(alpha)`
finite everywhere while remaining numerically indistinguishable from 1.0
for every other purpose (linear Euler stepping, the model's timestep
conditioning, etc). The rest of the schedule (single `shift`-warp of a
linear grid) is unchanged from `flow_match.py`, unlike the reference's own
`set_timesteps`, which re-derives the warp a second time in a way that is
numerically close enough to not matter here.

The reference is a stateful `nn.Module`-adjacent object: `self.model_outputs`,
`self.timestep_list`, `self.last_sample`, `self.lower_order_nums`, and
`self._step_index` are mutated in place across calls to `.step()`. JAX
prefers explicit, immutable state, so this port threads a `UniPCState`
dataclass through a pure `step(state, model_output, step_index, x) ->
(state, x)` function instead -- `step_index` plays the role the reference's
internal `self._step_index` counter plays, but the caller (an ordinary
eager Python sampling loop, exactly like `RectifiedFlowScheduler`'s
intended usage) owns and advances it.

Two bits of reference state are deliberately dropped:
  - `timestep_list`: read in `multistep_uni_p_bh_update` (`s0 =
    self.timestep_list[-1]`) but never actually used afterward -- dead
    code in the reference. Sigma/lambda lookups all key off `step_index`
    directly, which this port already threads through.
  - `lower_order_nums`: in the reference this is initialized to 0 and
    incremented by 1 every step until it saturates at `solver_order`; since
    `step()` is always called exactly once per `step_index` in strict
    increasing order, at the start of the call handling `step_index` it is
    always exactly `min(step_index, solver_order)` -- so it is recomputed
    from `step_index` on the fly here instead of carried in the state.

We only support the branches Cosmos2.5 actually exercises:
`predict_x0=True`, `prediction_type="flow_prediction"`, `solver_type="bh2"`,
no dynamic shifting (`use_dynamic_shifting=False`), no thresholding, and no
`solver_p`. Dead code for those disabled paths (`_threshold_sample`,
`_sigma_to_t`, `time_shift`, the `solver_p` branches) is not ported.
"""

import dataclasses
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp


def _sigma_to_alpha_sigma_t(sigma: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reference `_sigma_to_alpha_sigma_t`: flow-matching's alpha is simply `1 - sigma`."""
    return 1 - sigma, sigma


@dataclasses.dataclass(frozen=True)
class UniPCState:
    """Rolling predictor-corrector state threaded across `FlowUniPCMultistepScheduler.step` calls.

    Attributes:
        model_outputs: `(solver_order, *sample.shape)` history of *converted*
            (x0-prediction) model outputs, oldest first, newest at index -1.
            `None` until the first `step()` call, at which point it is
            lazily allocated (as all-zeros, then immediately overwritten in
            the freshly-added slot) once the sample shape is known --
            mirrors the reference's `self.model_outputs = [None] *
            solver_order` sentinel list, minus the unused early slots ever
            actually being read (a slot is never consulted at an order that
            would still see `None` there; see `this_order` below).
        last_sample: The sample *before* the most recent predictor step
            (i.e. after the most recent corrector step, or the raw input on
            the very first step) -- reference's `self.last_sample`. `None`
            before the first `step()` call, in which case the corrector is
            skipped (matching `self.last_sample is not None` in the
            reference's `use_corrector` condition).
        this_order: The predictor/corrector order used in the *upcoming*
            corrector call -- reference's `self.this_order`, which is
            computed at the bottom of one `step()` call and consumed at the
            top of the next.
    """

    model_outputs: Optional[jnp.ndarray]
    last_sample: Optional[jnp.ndarray]
    this_order: int


# A plain `@dataclass` is *not* automatically a JAX pytree -- passing one
# into a `jax.jit`-wrapped function raises ("as an abstract array... was not
# marked as static"). `model_outputs`/`last_sample` are genuine array data
# (or `None`, which JAX already treats as an empty pytree subtree, so the
# lazy-allocation-on-first-`step()` pattern above just works); `this_order`
# is a small Python int consumed by *Python-level* branching inside
# `multistep_uni_p_bh_update`/`_c_bh_update` (e.g. how many `D1s` terms to
# build) -- it has to be static at trace time regardless, so marking it a
# pytree "meta" (auxiliary, hashed-not-traced) field is both necessary for
# jit and correct: it takes only `solver_order`-many distinct values across
# a whole sampling run (the warmup ramp 1, 2, ..., solver_order), so this
# costs at most `solver_order` retraces per script invocation, not one per
# step.
jax.tree_util.register_dataclass(
    UniPCState, data_fields=["model_outputs", "last_sample"], meta_fields=["this_order"])


class FlowUniPCMultistepScheduler:
    """Flow-matching UniPC multistep predictor-corrector sampler. See module docstring."""

    def __init__(
        self,
        num_steps: int = 35,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        solver_order: int = 2,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: Sequence[int] = (),
    ):
        if solver_type != "bh2":
            # bh1 is supported by the reference too, but Cosmos2.5 (and every
            # caller we care about) uses bh2; keep the surface area small.
            raise NotImplementedError(f"solver_type={solver_type!r} not ported; only 'bh2' is supported")

        self.num_steps = num_steps
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.solver_order = solver_order
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.disable_corrector = tuple(disable_corrector)

        # Same style as RectifiedFlowScheduler (flow_match.py): linear sigma
        # grid warped once by `shift`. Starts at `1 - 1/num_train_timesteps`
        # rather than exactly `1.0` -- see module docstring; this is load
        # bearing for UniPC (unlike for Euler), not just cosmetic.
        sigma_max = 1.0 - 1.0 / num_train_timesteps
        sigmas = jnp.linspace(sigma_max, 0.0, num_steps + 1)
        if shift != 1.0:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.sigmas = sigmas.astype(jnp.float32)  # (num_steps + 1,): 1.0 -> 0.0, warped.
        self.timesteps = self.sigmas * num_train_timesteps  # for model conditioning only.

    def init_state(self) -> UniPCState:
        """Fresh state for a new sampling run. `this_order` is unused until step_index=1
        (the corrector is always skipped on the first call), so its initial value is
        irrelevant; solver_order is as good a placeholder as any."""
        return UniPCState(model_outputs=None, last_sample=None, this_order=self.solver_order)

    # -- predictor/corrector math -------------------------------------------------
    #
    # Both `multistep_uni_p_bh_update` (predictor) and
    # `multistep_uni_c_bh_update` (corrector) fit a degree-(order-1)
    # polynomial through `order` points of prior (converted) model output --
    # x0-predictions at "lambda" locations `lambda_si = log(alpha) -
    # log(sigma)` -- and use it to analytically integrate the (predict_x0)
    # exponential-integrator ODE from the current lambda to the target one.
    # `R @ rhos = b` solves for the polynomial-fit coefficients (`rhos_p`/
    # `rhos_c`) in closed form; order in {2 (predictor), 1 (corrector)} has
    # a hand-derived closed-form solution (`0.5`) instead of an explicit
    # solve, ported verbatim from the reference/upstream diffusers.

    def _coeffs(self, hh: jnp.ndarray, rks: jnp.ndarray, order: int):
        """Shared `h_phi_1`/`B_h`/`R`/`b` construction used by both predictor and
        corrector (reference lines 416-436 / 551-571, identical in both methods).
        `hh = -h` (since `predict_x0=True` always here); `rks` is the length-`order`
        vector of relative lambda offsets (last entry always 1.0)."""
        h_phi_1 = jnp.expm1(hh)  # h*phi_1(h) = e^h - 1
        B_h = jnp.expm1(hh)  # solver_type == "bh2"

        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1.0
        R_rows = []
        b_vals = []
        for i in range(1, order + 1):
            R_rows.append(jnp.power(rks, i - 1))
            b_vals.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        R = jnp.stack(R_rows)  # (order, order)
        b = jnp.stack(b_vals)  # (order,)
        return h_phi_1, B_h, R, b

    def multistep_uni_p_bh_update(
        self,
        sample: jnp.ndarray,
        model_outputs: jnp.ndarray,
        step_index: int,
        order: int,
    ) -> jnp.ndarray:
        """UniP predictor: extrapolate `x` at `sigmas[step_index + 1]` from `sample`
        (at `sigmas[step_index]`) and the last `order` converted model outputs.
        Reference: `multistep_uni_p_bh_update`, lines 337-464."""
        x = sample
        m0 = model_outputs[-1]

        sigma_t, sigma_s0 = self.sigmas[step_index + 1], self.sigmas[step_index]
        alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = _sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = jnp.log(alpha_t) - jnp.log(sigma_t)
        lambda_s0 = jnp.log(alpha_s0) - jnp.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            si = step_index - i
            mi = model_outputs[-(i + 1)]
            alpha_si, sigma_si = _sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = jnp.log(alpha_si) - jnp.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        rks.append(jnp.asarray(1.0))
        rks = jnp.stack(rks)

        hh = -h  # predict_x0 == True
        _, B_h, R, b = self._coeffs(hh, rks, order)
        h_phi_1 = jnp.expm1(hh)

        has_history = order > 1  # equivalent to `len(D1s) > 0` in the reference
        if has_history:
            D1s = jnp.stack(D1s, axis=0)  # (K, *sample.shape)
            if order == 2:
                rhos_p = jnp.asarray([0.5], dtype=x.dtype)
            else:
                rhos_p = jnp.linalg.solve(R[:-1, :-1], b[:-1]).astype(x.dtype)
            pred_res = jnp.einsum("k,k...->...", rhos_p, D1s)
        else:
            pred_res = 0.0

        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        x_t = x_t_ - alpha_t * B_h * pred_res
        return x_t.astype(x.dtype)

    def multistep_uni_c_bh_update(
        self,
        this_model_output: jnp.ndarray,
        last_sample: jnp.ndarray,
        this_sample: jnp.ndarray,
        model_outputs: jnp.ndarray,
        step_index: int,
        order: int,
    ) -> jnp.ndarray:
        """UniC corrector: refine `this_sample` (at `sigmas[step_index]`, already
        predicted from `last_sample` at `sigmas[step_index - 1]`) using the freshly
        computed `this_model_output`. Reference: `multistep_uni_c_bh_update`, lines
        466-601."""
        x = last_sample
        model_t = this_model_output
        m0 = model_outputs[-1]

        sigma_t, sigma_s0 = self.sigmas[step_index], self.sigmas[step_index - 1]
        alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = _sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = jnp.log(alpha_t) - jnp.log(sigma_t)
        lambda_s0 = jnp.log(alpha_s0) - jnp.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            si = step_index - (i + 1)
            mi = model_outputs[-(i + 1)]
            alpha_si, sigma_si = _sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = jnp.log(alpha_si) - jnp.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        rks.append(jnp.asarray(1.0))
        rks = jnp.stack(rks)

        hh = -h  # predict_x0 == True
        _, B_h, R, b = self._coeffs(hh, rks, order)
        h_phi_1 = jnp.expm1(hh)

        has_history = order > 1  # equivalent to `len(D1s) > 0` in the reference
        if has_history:
            D1s = jnp.stack(D1s, axis=0)
        if order == 1:
            rhos_c = jnp.asarray([0.5], dtype=x.dtype)
        else:
            rhos_c = jnp.linalg.solve(R, b).astype(x.dtype)

        corr_res = jnp.einsum("k,k...->...", rhos_c[:-1], D1s) if has_history else 0.0
        D1_t = model_t - m0

        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        return x_t.astype(x.dtype)

    def convert_model_output(self, sample: jnp.ndarray, model_output: jnp.ndarray, step_index: int) -> jnp.ndarray:
        """`predict_x0=True`, `prediction_type="flow_prediction"` branch of the
        reference's `convert_model_output` (lines 266-335): `x0 = x - sigma_t * v`."""
        sigma_t = self.sigmas[step_index]
        return sample - sigma_t * model_output

    def step(
        self,
        state: UniPCState,
        model_output: jnp.ndarray,
        step_index: int,
        x: jnp.ndarray,
    ) -> Tuple[UniPCState, jnp.ndarray]:
        """One UniPC predictor-corrector step from `step_index` to `step_index + 1`.

        Args:
            state: Rolling history state; use `self.init_state()` before the first call.
            model_output: The predicted velocity (v_t) from the DiT, at `step_index`.
            step_index: Index into `self.sigmas`/`self.timesteps` for the current step
                (0 <= step_index < num_steps). Caller-owned, like `RectifiedFlowScheduler`.
            x: Current latent state.

        Returns:
            `(new_state, new_x)`.
        """
        # Corrector uses the order that was decided at the end of the *previous*
        # step() call (reference: `order=self.this_order`, read before it's
        # overwritten later in the same call).
        use_corrector = (
            step_index > 0 and (step_index - 1) not in self.disable_corrector and state.last_sample is not None
        )

        model_output_convert = self.convert_model_output(x, model_output, step_index)

        if use_corrector:
            x = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=state.last_sample,
                this_sample=x,
                model_outputs=state.model_outputs,
                step_index=step_index,
                order=state.this_order,
            )

        # Push the newly converted output into the rolling history (drop oldest).
        if state.model_outputs is None:
            history = jnp.broadcast_to(model_output_convert, (self.solver_order,) + model_output_convert.shape)
            # Only the last slot is meaningfully populated; earlier (unused,
            # duplicate) slots are never read at an order low enough to reach
            # them -- see `this_order` computation below (warmup ramp).
        else:
            history = jnp.concatenate(
                [state.model_outputs[1:], model_output_convert[None]], axis=0
            )

        # lower_order_final ramp near the end of sampling (reference lines 689-694).
        if self.lower_order_final:
            this_order = min(self.solver_order, self.num_steps - step_index)
        else:
            this_order = self.solver_order
        # lower_order_nums warmup: reference's self.lower_order_nums is exactly
        # min(step_index, solver_order) at this point in the call (see module
        # docstring), so `lower_order_nums + 1` is `min(step_index, solver_order) + 1`.
        lower_order_nums = min(step_index, self.solver_order)
        this_order = min(this_order, lower_order_nums + 1)
        assert this_order > 0

        last_sample = x
        prev_sample = self.multistep_uni_p_bh_update(
            sample=x,
            model_outputs=history,
            step_index=step_index,
            order=this_order,
        )

        new_state = UniPCState(model_outputs=history, last_sample=last_sample, this_order=this_order)
        return new_state, prev_sample
