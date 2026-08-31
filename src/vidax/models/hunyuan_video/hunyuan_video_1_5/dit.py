"""HunyuanVideo-1.5 DiT (T2V+I2V unified) -- port of
``hyvideo/models/transformers/hunyuanvideo_1_5_transformer.py``'s
``HunyuanVideo_1_5_DiffusionTransformer.forward``.

Only the code path the 4 core (non-distilled, non-sparse) checkpoints
actually exercise is implemented: ``attn_mode="flash"`` semantics (see
``common/dit_layers.py``'s masking note), ``text_projection=
"single_refiner"``, ``guidance_embed=False``, ``text_pool_type=None`` (no
secondary pooled-text vector / ``vector_in``), no sequence-parallel, no
``output_features``/meanflow. These all match every one of the 4 real
``config.json``s read from the downloaded checkpoints (see ``configs.py``).

T2V vs I2V is **not** a different code path -- both call this same
``__call__`` with the same shapes; the caller (example script /
conditioning-prep util) is responsible for zeroing the reference-frame
channel-concat block and the vision-token stream for pure T2V, exactly as
the reference pipeline does (``hidden_states`` always has the
``concat_condition``-doubled+1 channel count; ``vision_states`` is always
passed, zeroed for T2V).
"""
from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from vidax.core.rope3d import sinusoidal_embedding_1d
from vidax.models.hunyuan_video.common.dit_layers import (
    FinalLayer,
    MLP,
    MMDoubleStreamBlock,
    MMSingleStreamBlock,
    SingleTokenRefiner,
    _timestep_embedder,
)
from vidax.models.hunyuan_video.common.rope import create_hunyuan_rope3d_freqs


def _patchify(x: jnp.ndarray, patch_size: Tuple[int, int, int]) -> Tuple[jnp.ndarray, Tuple[int, int, int]]:
    """(B, C, T, H, W) -> (B, t*h*w, C*pt*ph*pw), channel order (C, pt, ph, pw)
    flattened slowest-to-fastest -- matches a stride==kernel Conv3d's own
    weight-flatten order (see module-level design note in this file's
    docstring companion, ``configs.py``), so the translator can transpose
    the real Conv3d weight straight into a Dense kernel.
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


def _reorder_tokens(
    first: jnp.ndarray, second: jnp.ndarray,
    first_mask: jnp.ndarray, second_mask: jnp.ndarray,
    zero_feat: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Port of ``reorder_txt_token`` (``is_reorder=True`` path, the only one
    the reference DiT's ``forward`` ever calls).

    Concatenates ``[first; second]`` then, per batch item, stably reorders
    so all *valid* tokens (by mask) come first (in their original relative
    order, ``first`` before ``second``), followed by all invalid/padded
    tokens (same relative-order rule) -- exactly
    ``cat([first[mask], second[mask], first[~mask], second[~mask]])``.
    Implemented as a global stable sort keyed by (validity, original
    index): since ``first`` occupies the smaller original indices, this
    reproduces the reference's per-source grouping without needing a
    per-source gather (see the module this was ported into for the proof).
    """
    combined = jnp.concatenate([first, second], axis=1)
    combined_mask = jnp.concatenate([first_mask, second_mask], axis=1).astype(bool)
    if zero_feat:
        combined = jnp.where(combined_mask[..., None], combined, 0.0)
    n = combined.shape[1]
    cost = jnp.where(combined_mask, 0, 1)
    idx = jnp.broadcast_to(jnp.arange(n), cost.shape)
    order = jax.vmap(lambda c, i: jnp.lexsort((i, c)))(cost, idx)
    reordered = jnp.take_along_axis(combined, order[..., None], axis=1)
    reordered_mask = jnp.take_along_axis(combined_mask, order, axis=1)
    return reordered, reordered_mask.astype(jnp.int32)


class HunyuanVideo15DiT(nn.Module):
    patch_size: Tuple[int, int, int] = (1, 1, 1)
    in_channels: int = 32
    out_channels: int = 32
    concat_condition: bool = True
    is_reshape_temporal_channels: bool = False
    hidden_size: int = 2048
    heads_num: int = 16
    mlp_width_ratio: float = 4.0
    mlp_act_type: str = "gelu_tanh"
    mm_double_blocks_depth: int = 54
    mm_single_blocks_depth: int = 0
    rope_dim_list: Tuple[int, int, int] = (16, 56, 56)
    rope_theta: float = 256.0
    qkv_bias: bool = True
    qk_norm: bool = True
    guidance_embed: bool = False
    text_projection: str = "single_refiner"
    use_attention_mask: bool = True
    text_states_dim: int = 3584
    text_pool_type: Optional[str] = None
    text_states_dim_2: Optional[int] = None
    glyph_byT5_v2: bool = True
    vision_projection: str = "linear"
    vision_states_dim: int = 1152
    use_cond_type_embedding: bool = True
    mesh: Optional[Mesh] = None  # Megatron TP mesh. Threaded into every
    # block that calls masked_self_attention, including txt_in
    # (SingleTokenRefiner) -- once *any* part of the jitted program is
    # multi-device GSPMD-partitioned (true whenever double_blocks/
    # single_blocks' Q/K/V/output Dense layers carry TP sharding), *every*
    # Pallas/Mosaic flash-attention call anywhere in that same program
    # needs a `shard_map` wrapper, not just the ones whose own operands are
    # physically split (see `common/dit_layers.py`'s
    # `_flash_attention_tpu_segment_masked_replicated` docstring for the
    # real crash this fixes). txt_in's own weights are never TP-sharded
    # (its fused self_attn_qkv Dense is deliberately left un-sharded, see
    # docs/hardware_and_sharding.md), so it always gets the
    # `tp_sharded=False` (fully-replicated `shard_map`) path -- passing
    # `mesh` there is required for correctness under TP, not optional.

    def setup(self):
        assert self.text_projection == "single_refiner", "only single_refiner is ported"
        assert not self.is_reshape_temporal_channels, "not needed by any of the 4 core checkpoints"

        proj_in_channels = self.in_channels
        if self.concat_condition:
            proj_in_channels = self.in_channels * 2 + 1
        self.img_in_proj = nn.Dense(self.hidden_size, name="img_in_proj")
        self._proj_in_channels = proj_in_channels

        self.txt_in = SingleTokenRefiner(
            self.text_states_dim, self.hidden_size, self.heads_num, depth=2, mesh=self.mesh, name="txt_in")

        self.time_in = _timestep_embedder(self.hidden_size, name="time_in")
        self.vector_in = (
            _mlp_embedder(self.text_states_dim_2, self.hidden_size, name="vector_in")
            if self.text_pool_type is not None else None
        )
        self.guidance_in = (
            _timestep_embedder(self.hidden_size, name="guidance_in") if self.guidance_embed else None
        )

        if self.vision_projection == "linear":
            self.vision_in = _vision_projection(self.vision_states_dim, self.hidden_size, name="vision_in")
        else:
            self.vision_in = None

        if self.glyph_byT5_v2:
            from vidax.models.hunyuan_video.hunyuan_video_1_5.byt5 import ByT5Mapper
            self.byt5_in = ByT5Mapper(
                in_dim=1472, hidden_dim=2048, out_dim1=self.hidden_size, name="byt5_in")

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

        if self.use_cond_type_embedding:
            self.cond_type_embedding = nn.Embed(3, self.hidden_size, embedding_init=nn.initializers.zeros,
                                                 name="cond_type_embedding")
        else:
            self.cond_type_embedding = None

    def __call__(
        self,
        hidden_states: jnp.ndarray,          # (B, proj_in_channels, T, H, W)
        timestep: jnp.ndarray,               # (B,)
        text_states: jnp.ndarray,            # (B, L_txt, text_states_dim)
        encoder_attention_mask: jnp.ndarray,  # (B, L_txt)
        vision_states: Optional[jnp.ndarray] = None,      # (B, L_vis, vision_states_dim)
        byt5_text_states: Optional[jnp.ndarray] = None,   # (B, L_byt5, 1472)
        byt5_text_mask: Optional[jnp.ndarray] = None,      # (B, L_byt5)
        text_states_2: Optional[jnp.ndarray] = None,
        guidance: Optional[jnp.ndarray] = None,
        mask_type: str = "t2v",
    ) -> jnp.ndarray:
        bs, _, ot, oh, ow = hidden_states.shape
        pt, ph, pw = self.patch_size
        tt, th, tw = ot // pt, oh // ph, ow // pw

        freqs = create_hunyuan_rope3d_freqs(tt, th, tw, self.rope_dim_list, self.rope_theta)

        img, _ = _patchify(hidden_states, self.patch_size)
        img = self.img_in_proj(img)

        vec = self.time_in(timestep)
        if self.vector_in is not None and text_states_2 is not None:
            vec = vec + self.vector_in(text_states_2)
        if self.guidance_in is not None:
            assert guidance is not None
            vec = vec + self.guidance_in(guidance)

        text_mask = encoder_attention_mask
        txt = self.txt_in(text_states, timestep, text_mask if self.use_attention_mask else None)

        if self.cond_type_embedding is not None:
            txt = txt + self.cond_type_embedding(jnp.zeros(txt.shape[:2], dtype=jnp.int32))

        if self.glyph_byT5_v2:
            byt5_txt = self.byt5_in(byt5_text_states)
            if self.cond_type_embedding is not None:
                byt5_txt = byt5_txt + self.cond_type_embedding(jnp.ones(byt5_txt.shape[:2], dtype=jnp.int32))
            txt, text_mask = _reorder_tokens(byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=True)

        if self.vision_in is not None and vision_states is not None:
            extra = self.vision_in(vision_states)
            if mask_type == "t2v":
                extra_mask = jnp.zeros(extra.shape[:2], dtype=text_mask.dtype)
                extra = extra * 0.0
            else:
                extra_mask = jnp.ones(extra.shape[:2], dtype=text_mask.dtype)
            if self.cond_type_embedding is not None:
                extra = extra + self.cond_type_embedding(jnp.full(extra.shape[:2], 2, dtype=jnp.int32))
            txt, text_mask = _reorder_tokens(extra, txt, extra_mask, text_mask, zero_feat=False)

        img_len = img.shape[1]
        key_valid = jnp.concatenate(
            [jnp.ones((bs, img_len), dtype=bool), text_mask.astype(bool)], axis=1)

        for block in self.double_blocks:
            img, txt = block(img, txt, vec, freqs, key_valid)

        txt_len = txt.shape[1]
        x = jnp.concatenate([img, txt], axis=1)
        for block in self.single_blocks:
            x = block(x, vec, txt_len, freqs, key_valid)
        img = x[:, :img_len]

        img = self.final_layer(img, vec)
        return _unpatchify(img, (tt, th, tw), self.patch_size, self.out_channels)


def _mlp_embedder(in_dim: int, hidden_dim: int, name: str):
    class _MLPEmbedder(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(hidden_dim, name="in_layer")(x)
            x = nn.silu(x)
            return nn.Dense(hidden_dim, name="out_layer")(x)
    return _MLPEmbedder(name=name)


def _vision_projection(in_dim: int, out_dim: int, name: str):
    """``VisionProjection``: LN(in) -> Linear(in,in) -> GELU -> Linear(in,out) -> LN(out)."""
    class _VisionProjection(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.LayerNorm(epsilon=1e-5, name="ln_0")(x)
            x = nn.Dense(in_dim, name="linear_1")(x)
            x = nn.gelu(x, approximate=False)
            x = nn.Dense(out_dim, name="linear_3")(x)
            x = nn.LayerNorm(epsilon=1e-5, name="ln_4")(x)
            return x
    return _VisionProjection(name=name)
