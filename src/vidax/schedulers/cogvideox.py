"""CogVideoX schedulers (Flax/JAX-friendly, no PyTorch dependency).

Structural ports of diffusers' `CogVideoXDDIMScheduler` and
`CogVideoXDPMScheduler` (`diffusers/schedulers/scheduling_{ddim,dpm}_cogvideox.py`).
Every released CogVideoX checkpoint ships `prediction_type="v_prediction"`,
`beta_schedule="scaled_linear"` (0.00085..0.0120), `timestep_spacing="trailing"`,
`rescale_betas_zero_snr=True`, `set_alpha_to_one=True`; only `snr_shift_scale`
differs (3.0 for CogVideoX-2b, 1.0 for everything else -- see
`vidax.models.cogvideo.configs`).

The `use_dynamic_cfg` guidance-scale schedule lives in the example script
(`examples/generate_cogvideox.py`), matching where the diffusers *pipeline*
puts it -- it is not part of the scheduler.

`alphas_cumprod` is a static host-side numpy array; `step` looks up the
integer-timestep coefficients on host and applies them to the jnp `sample`,
so the per-step call stays cheap and the example can keep a plain Python
denoising loop (like `examples/generate_ltx_video.py`).
"""
import numpy as np


def _make_alphas_cumprod(num_train_timesteps, beta_start, beta_end, snr_shift_scale, rescale_zero_snr):
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=np.float64) ** 2
    alphas = 1.0 - betas
    ac = np.cumprod(alphas, axis=0)
    # SD3-style SNR shift.
    ac = ac / (snr_shift_scale + (1 - snr_shift_scale) * ac)
    if rescale_zero_snr:
        abar_sqrt = np.sqrt(ac)
        s0, sT = abar_sqrt[0].copy(), abar_sqrt[-1].copy()
        abar_sqrt = abar_sqrt - sT
        abar_sqrt = abar_sqrt * (s0 / (s0 - sT))
        ac = abar_sqrt ** 2
    return ac  # float64


def _trailing_timesteps(num_train_timesteps, num_inference_steps):
    step_ratio = num_train_timesteps / num_inference_steps
    ts = np.round(np.arange(num_train_timesteps, 0, -step_ratio)).astype(np.int64)
    ts -= 1
    return ts


class _CogVideoXBaseScheduler:
    def __init__(
        self,
        num_inference_steps: int,
        *,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
        snr_shift_scale: float = 1.0,
        set_alpha_to_one: bool = True,
        rescale_betas_zero_snr: bool = True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.alphas_cumprod = _make_alphas_cumprod(
            num_train_timesteps, beta_start, beta_end, snr_shift_scale, rescale_betas_zero_snr)
        self.final_alpha_cumprod = 1.0 if set_alpha_to_one else float(self.alphas_cumprod[0])
        self.init_noise_sigma = 1.0
        self.timesteps = _trailing_timesteps(num_train_timesteps, num_inference_steps)

    def _alpha(self, t):
        # np.float64 (not python float): downstream `1 - a` can be exactly 0
        # with zero-terminal-SNR / `set_alpha_to_one`, and numpy division by
        # zero yields inf (which the DPM math absorbs) where python raises.
        return np.float64(self.alphas_cumprod[t])

    def _alpha_prev(self, prev_t):
        return np.float64(self.alphas_cumprod[prev_t]) if prev_t >= 0 else np.float64(self.final_alpha_cumprod)

    def _pred_original_sample(self, sample, model_output, alpha_prod_t):
        # v_prediction
        beta_prod_t = 1.0 - alpha_prod_t
        return (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output


class CogVideoXDDIMScheduler(_CogVideoXBaseScheduler):
    """Deterministic (eta=0) DDIM update for CogVideoX's v-prediction model."""

    def step(self, model_output, timestep: int, sample):
        prev_t = int(timestep) - self.num_train_timesteps // self.num_inference_steps
        a_t = self._alpha(int(timestep))
        a_prev = self._alpha_prev(prev_t)

        pred_x0 = self._pred_original_sample(sample, model_output, a_t)
        coef_a = ((1.0 - a_prev) / (1.0 - a_t)) ** 0.5
        coef_b = a_prev ** 0.5 - a_t ** 0.5 * coef_a
        prev_sample = coef_a * sample + coef_b * pred_x0
        return prev_sample, pred_x0


class CogVideoXDPMScheduler(_CogVideoXBaseScheduler):
    """DPM-Solver++ (multistep) update, matching `CogVideoXDPMScheduler.step`.

    `timestep_back` is the previous step's timestep (`None` on the first
    step). `noise` / `noise2` are caller-supplied Gaussian samples the same
    shape as `sample` -- the reference draws fresh noise for both the
    single-step prediction and (when advancing) the multistep correction.
    """

    @staticmethod
    def _variables(a_t, a_prev, a_back):
        # `np.log` (not `math.log`): with zero-terminal-SNR, `alphas_cumprod`
        # at the first timestep is exactly 0, so `lamb` is -inf -- torch's
        # `.log()` produces -inf here too, and the downstream `exp(-h)` /
        # `expm1(-2h)` collapse it back to a finite update. `math.log(0)`
        # would instead raise.
        with np.errstate(divide="ignore", invalid="ignore"):
            lamb = np.log((a_t / (1.0 - a_t)) ** 0.5)
            lamb_next = np.log((a_prev / (1.0 - a_prev)) ** 0.5)
            h = lamb_next - lamb
            if a_back is not None:
                lamb_prev = np.log((a_back / (1.0 - a_back)) ** 0.5)
                return h, (lamb - lamb_prev) / h
        return h, None

    @staticmethod
    def _mult(h, r, a_t, a_prev, a_back):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mult1 = ((1.0 - a_prev) / (1.0 - a_t)) ** 0.5 * np.exp(-h)
            mult2 = np.expm1(-2.0 * h) * a_prev ** 0.5
            if a_back is not None:
                # r == 0 (two consecutive timesteps -> same lambda, possible at
                # the zero-SNR tail) makes these inf; the reference's torch math
                # produces inf here too, and this multistep-correction branch is
                # only taken when `old_pred_original_sample` exists (not step 0).
                return mult1, mult2, 1.0 + 1.0 / (2.0 * r), 1.0 / (2.0 * r)
        return mult1, mult2

    def step(self, model_output, old_pred_original_sample, timestep: int, timestep_back,
             sample, noise, noise2=None):
        prev_t = int(timestep) - self.num_train_timesteps // self.num_inference_steps
        a_t = self._alpha(int(timestep))
        a_prev = self._alpha_prev(prev_t)
        a_back = self._alpha(int(timestep_back)) if timestep_back is not None else None

        pred_x0 = self._pred_original_sample(sample, model_output, a_t)
        h, r = self._variables(a_t, a_prev, a_back)
        mult = self._mult(h, r, a_t, a_prev, a_back)
        with np.errstate(over="ignore", invalid="ignore"):
            mult_noise = (1.0 - a_prev) ** 0.5 * (1.0 - np.exp(-2.0 * h)) ** 0.5

        prev_sample = mult[0] * sample - mult[1] * pred_x0 + mult_noise * noise
        if old_pred_original_sample is None or prev_t < 0:
            return prev_sample, pred_x0

        denoised_d = mult[2] * pred_x0 - mult[3] * old_pred_original_sample
        n2 = noise if noise2 is None else noise2
        x_advanced = mult[0] * sample - mult[1] * denoised_d + mult_noise * n2
        return x_advanced, pred_x0
