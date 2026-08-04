import jax.numpy as jnp


class RectifiedFlowScheduler:
    """
    Euler sampler for Rectified Flow / Flow Matching models, matching the
    reference `FlowUniPCMultistepScheduler`/`FlowDPMSolverMultistepScheduler`
    noise schedule (Wan2.1-main/wan/utils/fm_solvers_unipc.py's
    `set_timesteps`), minus its higher-order multistep predictor-corrector
    integration (Euler is used here instead -- see vidax's README).

    Two related but distinct quantities matter here:
      - `sigmas`: the actual flow-matching interpolation coefficient, in
        [0, 1] (1 = pure noise, 0 = clean data). The Euler update
        `x_new = x - v * dsigma` operates in this space.
      - `timesteps`: `sigmas * num_train_timesteps`, i.e. on the same
        ~[0, 1000] scale as training. This is what actually gets fed to the
        DiT's sinusoidal timestep embedding -- passing raw `sigmas` there
        instead (as if `num_train_timesteps == 1`) feeds the model a
        conditioning signal about 1000x smaller than anything it saw during
        training, i.e. "barely any noise" at every step regardless of the
        true noise level, which produces incoherent output no matter how
        many sampling steps are taken.

    `shift` implements the same resolution-dependent noise-schedule warp
    Wan2.1 uses by default (5.0 for text-to-video): `sigma' = shift * sigma
    / (1 + (shift - 1) * sigma)`, which biases more of the schedule towards
    high-noise sigmas -- flow-matching models need relatively more of their
    sampling budget spent there as resolution (and thus token count) grows.
    """

    def __init__(self, num_steps: int = 50, num_train_timesteps: int = 1000, shift: float = 5.0):
        self.num_steps = num_steps
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

        sigmas = jnp.linspace(1.0, 0.0, num_steps + 1)
        if shift != 1.0:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.sigmas = sigmas  # (num_steps + 1,): 1.0 -> 0.0, warped.
        self.timesteps = sigmas * num_train_timesteps  # for model conditioning only.

    def step(
        self,
        model_output: jnp.ndarray,
        step_index: int,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Performs a single Euler step from `step_index` to `step_index + 1`.

        Args:
            model_output: The predicted velocity (v_t) from the DiT.
            step_index: Index into `self.sigmas`/`self.timesteps` for the
                current step (0 <= step_index < num_steps).
            x: Current latent state.

        Returns:
            jnp.ndarray: The next latent state.
        """
        dsigma = self.sigmas[step_index] - self.sigmas[step_index + 1]
        return x - model_output * dsigma.astype(x.dtype)
