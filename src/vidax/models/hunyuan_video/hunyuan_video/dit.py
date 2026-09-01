"""HunyuanVideo 1.0 DiT (T2V only) -- port of
``hyvideo/modules/models.py``'s ``HYVideoDiffusionTransformer.forward``.

See ``/home/congyued/.claude/plans/cryptic-weaving-bentley.md``'s
"Architecture diff: HunyuanVideo 1.0 vs 1.5" section for the full
component-by-component comparison against the already-ported
``hunyuan_video1_5`` DiT this reuses ``common/dit_layers.py``/
``common/rope.py`` from.

Only ``text_projection="single_refiner"`` (the only mode the reference ever
exercises for its released checkpoints) is implemented. I2V is explicitly
out of scope for this batch (separate, un-cloned upstream repo) -- there is
no ``concat_condition``/channel-doubling path here at all, unlike
``hunyuan_video1_5``'s unified T2V+I2V ``forward``.
"""
from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
from jax.sharding import Mesh

from vidax.models.hunyuan_video.common.dit_layers import (
    FinalLayer,
    MMDoubleStreamBlock,
    MMSingleStreamBlock,
    SingleTokenRefiner,
    _timestep_embedder,
)
from vidax.models.hunyuan_video.common.rope import create_hunyuan_rope3d_freqs


def _patchify(x: jnp.ndarray, patch_size: Tuple[int, int, int]) -> Tuple[jnp.ndarray, Tuple[int, int, int]]:
    """(B, C, T, H, W) -> (B, t*h*w, C*pt*ph*pw).

    Duplicated from ``hunyuan_video1_5/dit.py`` rather than imported --
    see the plan's architecture-diff section for why this reshape-based
    form is exact for any ``patch_size`` (a stride==kernel ``Conv3d`` has no
    patch overlap), not just 1.5's degenerate ``(1,1,1)`` case, and why it's
    duplicated rather than shared to avoid any cross-package coupling.
    """
    pt, ph, pw = patch_size
    b, c, T, H, W = x.shape
    t, h, w = T // pt, H // ph, W // pw
    x = x.reshape(b, c, t, pt, h, ph, w, pw)
    x = jnp.einsum("bctohpwq->bthwcopq", x)
    x = x.reshape(b, t * h * w, c * pt * ph * pw)
    return x, (t, h, w)


def _unpatchify(x: jnp.ndarray, thw: Tuple[int, int, int], patch_size: Tuple[int, int, int], out_channels: int) -> jnp.ndarray:
    """Inverse of ``_patchify`` -- matches the reference's ``unpatchify`` exactly."""
    t, h, w = thw
    pt, ph, pw = patch_size
    b = x.shape[0]
    x = x.reshape(b, t, h, w, out_channels, pt, ph, pw)
    x = jnp.einsum("bthwcopq->bctohpwq", x)
    return x.reshape(b, out_channels, t * pt, h * ph, w * pw)


def _mlp_embedder(hidden_dim: int, name: str):
    """``MLPEmbedder``: Dense(in,hidden) -> SiLU -> Dense(hidden,hidden),
    no sinusoidal step -- port of ``mlp_layers.py:MLPEmbedder``, used only
    for ``vector_in`` (raw pooled CLIP-L text vector, not a timestep). Small
    and single-caller, so defined locally rather than added to
    ``common/dit_layers.py``.
    """
    class _MLPEmbedder(nn.Module):
        @nn.compact
        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            x = nn.Dense(hidden_dim, name="in_layer")(x)
            x = nn.silu(x)
            return nn.Dense(hidden_dim, name="out_layer")(x)

    return _MLPEmbedder(name=name)


class HunyuanVideoDiT(nn.Module):
    """``HYVideoDiffusionTransformer``, T2V only.

    Text conditioning: a single LLM text encoder's refined hidden states
    (``txt_in``, ``SingleTokenRefiner``) feed the joint double/single-stream
    attention as ``txt``; a separate pooled CLIP-L vector (``vector_in``)
    feeds the AdaLN modulation vector ``vec`` alongside the timestep. No
    byT5/SigLIP/token-concatenation/``cond_type_embedding`` (1.5-only, see
    the plan's architecture diff).
    """
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    in_channels: int = 16
    out_channels: int = 16
    hidden_size: int = 3072
    heads_num: int = 24
    mlp_width_ratio: float = 4.0
    mlp_act_type: str = "gelu_tanh"
    mm_double_blocks_depth: int = 20
    mm_single_blocks_depth: int = 40
    rope_dim_list: Tuple[int, int, int] = (16, 56, 56)
    rope_theta: float = 256.0
    qkv_bias: bool = True
    qk_norm: bool = True
    guidance_embed: bool = False
    text_projection: str = "single_refiner"
    use_attention_mask: bool = True
    text_states_dim: int = 4096
    text_states_dim_2: int = 768
    i2v_condition_type: Optional[str] = None  # None (T2V) or "token_replace" -- see
    # `pre_process`'s docstring. HunyuanVideo-I2V's other mode, "latent_concat"
    # (channel-doubled img_in_proj, like 1.5's I2V), is never used by the
    # released `hunyuan-video-i2v-720p` checkpoint (its own default/only
    # shipped config is "token_replace" -- confirmed against
    # `hyvideo/config.py`'s `--i2v-condition-type` default), so it isn't
    # ported here.
    mesh: Optional[Mesh] = None  # Megatron TP mesh -- see
    # `hunyuan_video1_5/dit.py`'s identical field docstring for why `mesh`
    # must also be threaded into `txt_in` (a `SingleTokenRefiner`, always
    # `tp_sharded=False`) once *any* part of the jitted program is
    # multi-device GSPMD-partitioned.

    def setup(self):
        assert self.text_projection == "single_refiner", "only single_refiner is ported"

        self.img_in_proj = nn.Dense(self.hidden_size, name="img_in_proj")

        self.txt_in = SingleTokenRefiner(
            self.text_states_dim, self.hidden_size, self.heads_num, depth=2,
            mesh=self.mesh, name="txt_in")

        self.time_in = _timestep_embedder(self.hidden_size, name="time_in")
        self.vector_in = _mlp_embedder(self.hidden_size, name="vector_in")
        self.guidance_in = (
            _timestep_embedder(self.hidden_size, name="guidance_in") if self.guidance_embed else None
        )

        self.double_blocks = [
            MMDoubleStreamBlock(
                self.hidden_size, self.heads_num, self.mlp_width_ratio, self.mlp_act_type,
                self.qk_norm, self.qkv_bias, mesh=self.mesh, name=f"double_blocks_{i}")
            for i in range(self.mm_double_blocks_depth)
        ]
        self.single_blocks = [
            MMSingleStreamBlock(
                self.hidden_size, self.heads_num, self.mlp_width_ratio, self.mlp_act_type,
                self.qk_norm, mesh=self.mesh, name=f"single_blocks_{i}")
            for i in range(self.mm_single_blocks_depth)
        ]

        out_dim = 1
        for p in self.patch_size:
            out_dim *= p
        out_dim *= self.out_channels
        self.final_layer = FinalLayer(self.hidden_size, out_dim, name="final_layer")

    def pre_process(
        self,
        hidden_states: jnp.ndarray,          # (B, in_channels, T, H, W)
        timestep: jnp.ndarray,               # (B,)
        text_states: jnp.ndarray,            # (B, L_txt, text_states_dim) -- LLM hidden states
        encoder_attention_mask: jnp.ndarray,  # (B, L_txt)
        text_states_2: jnp.ndarray,          # (B, text_states_dim_2) -- pooled CLIP-L vector
        guidance: Optional[jnp.ndarray] = None,
    ):
        """Everything before the double-stream block loop: patchify,
        timestep/text embedding, and the token-refiner. Split out from
        `__call__` so `--offload_dit_weights` can run this once, then loop
        the (offloaded) double/single blocks externally, then call
        `mid_process`/`post_process` -- see
        `examples/generate_hunyuan_video.py` and docs/weight_offloading.md.

        Returns `(img, txt, vec, freqs, key_valid, img_len, tt, th, tw,
        token_replace_vec, first_frame_token_num)` -- `img_len`/`tt`/`th`/
        `tw` are only needed by `mid_process`/`post_process`, and the last
        two only by the block loops under I2V's `token_replace` mode
        (`None` otherwise), not by `mid_process`/`post_process`.

        Under `i2v_condition_type="token_replace"`, the caller (see
        `examples/generate_hunyuan_video.py`'s I2V path) is responsible for
        substituting the first *latent* frame of `hidden_states` with the
        reference image's own clean, un-noised VAE-encoded latent before
        every call here -- this method only computes the extra as-if-t=0
        modulation vector (`token_replace_vec`, matching the reference's
        `self.time_in(torch.zeros_like(t))`) that tells the block loop to
        modulate exactly that first frame's tokens differently from the
        rest; it does not itself touch `hidden_states`.
        """
        bs, _, ot, oh, ow = hidden_states.shape
        pt, ph, pw = self.patch_size
        tt, th, tw = ot // pt, oh // ph, ow // pw

        freqs = create_hunyuan_rope3d_freqs(tt, th, tw, self.rope_dim_list, self.rope_theta)

        img, _ = _patchify(hidden_states, self.patch_size)
        img = self.img_in_proj(img)

        token_replace_vec = None
        first_frame_token_num = None
        if self.i2v_condition_type == "token_replace":
            token_replace_vec = self.time_in(jnp.zeros_like(timestep))
            first_frame_token_num = th * tw

        vec_2 = self.vector_in(text_states_2)
        vec = self.time_in(timestep) + vec_2
        if token_replace_vec is not None:
            token_replace_vec = token_replace_vec + vec_2
        if self.guidance_in is not None:
            assert guidance is not None, "guidance_embed=True requires a `guidance` input"
            vec = vec + self.guidance_in(guidance)

        text_mask = encoder_attention_mask
        txt = self.txt_in(text_states, timestep, text_mask if self.use_attention_mask else None)

        img_len = img.shape[1]
        key_valid = jnp.concatenate(
            [jnp.ones((bs, img_len), dtype=bool), text_mask.astype(bool)], axis=1)

        return img, txt, vec, freqs, key_valid, img_len, tt, th, tw, token_replace_vec, first_frame_token_num

    @staticmethod
    def mid_process(img: jnp.ndarray, txt: jnp.ndarray):
        """Between the double-stream and single-stream block loops: the
        reference concatenates `[img; txt]` into one joint stream once the
        double-stream blocks (which keep `img`/`txt` separate) are done.
        Trivial, but its own step so `--offload_dit_weights` can call it
        between the two offloaded block loops without needing any params.
        """
        txt_len = txt.shape[1]
        return jnp.concatenate([img, txt], axis=1), txt_len

    def post_process(self, x: jnp.ndarray, vec: jnp.ndarray, img_len: int, tt: int, th: int, tw: int) -> jnp.ndarray:
        """Everything after the single-stream block loop: slice back out
        the image tokens, head projection, unpatchify. See `pre_process`'s
        docstring for why this is split out.
        """
        img = x[:, :img_len]
        img = self.final_layer(img, vec)
        return _unpatchify(img, (tt, th, tw), self.patch_size, self.out_channels)

    def __call__(
        self,
        hidden_states: jnp.ndarray,          # (B, in_channels, T, H, W)
        timestep: jnp.ndarray,               # (B,)
        text_states: jnp.ndarray,            # (B, L_txt, text_states_dim) -- LLM hidden states
        encoder_attention_mask: jnp.ndarray,  # (B, L_txt)
        text_states_2: jnp.ndarray,          # (B, text_states_dim_2) -- pooled CLIP-L vector
        guidance: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        img, txt, vec, freqs, key_valid, img_len, tt, th, tw, token_replace_vec, first_frame_token_num = self.pre_process(
            hidden_states, timestep, text_states, encoder_attention_mask, text_states_2, guidance=guidance)

        for block in self.double_blocks:
            img, txt = block(img, txt, vec, freqs, key_valid, token_replace_vec, first_frame_token_num)

        x, txt_len = self.mid_process(img, txt)
        for block in self.single_blocks:
            x = block(x, vec, txt_len, freqs, key_valid, token_replace_vec, first_frame_token_num)

        return self.post_process(x, vec, img_len, tt, th, tw)
