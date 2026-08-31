"""CogVideoX Diffusion Transformer backbone (Flax/JAX) -- covers every
released checkpoint (2b / 5b / 5b-I2V / 1.5-5B / 1.5-5B-I2V) config-driven.

A structural port of `CogVideoXTransformer3DModel` / `CogVideoXBlock` from
`diffusers/models/transformers/cogvideox_transformer_3d.py` (+ the diffusers
`CogVideoXLayerNormZero` / `AdaLayerNorm` / `CogVideoXPatchEmbed` /
`CogVideoXAttnProcessor2_0` pieces it composes). Submodule names mirror the
PyTorch module names one-for-one so
`vidax.translator.mappings.cogvideox.map_cogvideox_dit_keys` stays a
near-mechanical prefix-strip.

Architectural notes not obvious from the reference:

- The 226 text tokens and the F*H/p*W/p visual tokens are concatenated into
  **one joint self-attention sequence** per block; there is no cross-
  attention. `CogVideoXLayerNormZero` modulates the two halves with
  *separate* shift/scale/gate triples (`shift/scale/gate` for visual,
  `enc_*` for text) sharing one `LayerNorm`. The FFN likewise runs once over
  the re-concatenated `[text; visual]` sequence.
- RoPE (`vidax.models.cogvideo.rope`) is per-head over `attention_head_dim`
  (64) and applied to the **visual slice of q and k only** -- text tokens
  are never rotated (diffusers `CogVideoXAttnProcessor2_0`).
- `norm_final` (plain LayerNorm) is applied to the visual tokens only after
  the block loop -- numerically identical to the older `cat([text,visual])`
  -> LayerNorm -> slice form since LayerNorm is per-token.
- 2b has no RoPE: it adds a fixed 3D sincos positional embedding
  (`rope.get_3d_sincos_pos_embed`, recomputed -- not a checkpoint buffer)
  to the joint sequence. 5b-I2V adds a *learned* positional embedding
  (a persistent `pos_embedding` buffer in the checkpoint). 1.5 uses
  `patch_size_t=2` temporal patchifying and a linear (not conv) patch proj.
"""
from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from vidax.core.attention import (
    chunk_by_rank, dot_product_attention, sequence_parallel_joint_self_attention,
)
from vidax.models.cogvideo.rope import apply_rotary_emb, get_3d_sincos_pos_embed

TEXT_SEQ_LENGTH = 226
ATTENTION_HEAD_DIM = 64
TIME_EMBED_DIM = 512
TEXT_EMBED_DIM = 4096
PATCH_SIZE = 2


def _timestep_sinusoidal_embedding(timesteps: jnp.ndarray, dim: int, max_period: int = 10000) -> jnp.ndarray:
    """diffusers `get_timestep_embedding(timesteps, dim, flip_sin_to_cos=True,
    downscale_freq_shift=0, scale=1, max_period=10000)` -- `timesteps` is 1D.
    """
    half = dim // 2
    exponent = -jnp.log(float(max_period)) * jnp.arange(half, dtype=jnp.float32) / half
    emb = jnp.exp(exponent)
    emb = timesteps.astype(jnp.float32)[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    # flip_sin_to_cos: [sin, cos] -> [cos, sin]
    emb = jnp.concatenate([emb[:, half:], emb[:, :half]], axis=-1)
    return emb


def _layernorm(x, scale, bias, eps):
    """torch `nn.LayerNorm` (elementwise_affine=True), computed in float32."""
    orig = x.dtype
    xf = x.astype(jnp.float32)
    mean = jnp.mean(xf, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
    normed = (xf - mean) * jax.lax.rsqrt(var + eps)
    normed = normed * scale.astype(jnp.float32) + bias.astype(jnp.float32)
    return normed.astype(orig)


class CogVideoXLayerNormZero(nn.Module):
    """diffusers `CogVideoXLayerNormZero`: `silu -> Linear(512, 6*dim)`, split
    into (shift, scale, gate, enc_shift, enc_scale, enc_gate); shared
    `LayerNorm(dim, eps=1e-5)` (affine) applied to the visual and text halves
    separately, each modulated by its own (shift, scale); the two gates are
    returned for the caller's residual add.
    """
    dim: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, hidden_states, encoder_hidden_states, temb):
        params6 = nn.Dense(6 * self.dim, name="linear")(nn.silu(temb))
        shift, scale, gate, enc_shift, enc_scale, enc_gate = jnp.split(params6, 6, axis=-1)
        ln_scale = self.param("norm_scale", nn.initializers.ones, (self.dim,))
        ln_bias = self.param("norm_bias", nn.initializers.zeros, (self.dim,))

        h = _layernorm(hidden_states, ln_scale, ln_bias, self.eps)
        h = h * (1 + scale[:, None, :]) + shift[:, None, :]
        e = _layernorm(encoder_hidden_states, ln_scale, ln_bias, self.eps)
        e = e * (1 + enc_scale[:, None, :]) + enc_shift[:, None, :]
        return h, e, gate[:, None, :], enc_gate[:, None, :]


class CogVideoXAttention(nn.Module):
    """diffusers `Attention` + `CogVideoXAttnProcessor2_0` for one block's
    `attn1`: joint self-attention over `[text(226); visual]`, per-head
    LayerNorm QK-norm (eps 1e-6), partial RoPE on the visual slice of q/k.
    """
    dim: int
    num_heads: int
    head_dim: int = ATTENTION_HEAD_DIM
    qk_eps: float = 1e-6
    mesh: Any = None
    sequence_parallel: bool = False
    sp_axis_name: str = "sp"

    @nn.compact
    def __call__(self, hidden_states, encoder_hidden_states, rope_cos, rope_sin):
        b = hidden_states.shape[0]
        text_len = encoder_hidden_states.shape[1]
        x = jnp.concatenate([encoder_hidden_states, hidden_states], axis=1)  # (B, 226+Nv, dim)
        s = x.shape[1]

        # `to_q/to_k/to_v` are column-parallel and `to_out_0` row-parallel
        # (already in `vidax.core.sharding`'s name lists, shared with LTX):
        # under `--tensor_parallel_size > 1` GSPMD splits the head axis of the
        # full-width Dense automatically -- no manual `dim_local` here.
        q = nn.Dense(self.dim, name="to_q")(x).reshape(b, s, self.num_heads, self.head_dim)
        k = nn.Dense(self.dim, name="to_k")(x).reshape(b, s, self.num_heads, self.head_dim)
        v = nn.Dense(self.dim, name="to_v")(x).reshape(b, s, self.num_heads, self.head_dim)

        # QK-norm is a per-head LayerNorm over head_dim (eps 1e-6) -- same for
        # every head, so its params stay replicated (not head-split).
        nq_s = self.param("norm_q_scale", nn.initializers.ones, (self.head_dim,))
        nq_b = self.param("norm_q_bias", nn.initializers.zeros, (self.head_dim,))
        nk_s = self.param("norm_k_scale", nn.initializers.ones, (self.head_dim,))
        nk_b = self.param("norm_k_bias", nn.initializers.zeros, (self.head_dim,))
        q = _layernorm(q, nq_s, nq_b, self.qk_eps)
        k = _layernorm(k, nk_s, nk_b, self.qk_eps)

        if rope_cos is not None:
            q_txt, q_vis = q[:, :text_len], q[:, text_len:]
            k_txt, k_vis = k[:, :text_len], k[:, text_len:]
            q = jnp.concatenate([q_txt, apply_rotary_emb(q_vis, rope_cos, rope_sin)], axis=1)
            k = jnp.concatenate([k_txt, apply_rotary_emb(k_vis, rope_cos, rope_sin)], axis=1)

        if self.sequence_parallel:
            # DeepSpeed-Ulysses: the visual tokens are sequence-chunked across
            # the 'sp' mesh axis (by `CogVideoXDiT.pre_process`); this reshuffles
            # to a head-sharded, full-sequence view just for the joint
            # `[text; visual]` attention itself. Must run inside `shard_map`
            # over `self.mesh` -- see `examples/generate_cogvideox.py`.
            out = sequence_parallel_joint_self_attention(
                q, k, v, text_len, self.sp_axis_name)
        elif self.mesh is not None:
            # TPU: real O(S) flash attention (materializing the (B, H, S, S)
            # matrix over ~1.8e4 tokens OOMs even at tp=4).
            out = dot_product_attention(q, k, v, mesh=self.mesh)
        else:
            scale = self.head_dim ** -0.5
            logits = jnp.einsum("bqhd,bkhd->bhqk", q, k).astype(jnp.float32) * scale
            weights = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
            out = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
        out = out.reshape(b, s, self.dim)

        out = nn.Dense(self.dim, name="to_out_0")(out)
        return out[:, text_len:], out[:, :text_len]  # visual, text


class CogVideoXFeedForward(nn.Module):
    """diffusers `FeedForward(activation_fn="gelu-approximate")`:
    `net.0.proj` (Linear + tanh-approx GELU) -> `net.2` (Linear back down).
    """
    dim: int
    inner_dim: int

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(self.inner_dim, name="net_0_proj")(x)
        h = nn.gelu(h, approximate=True)
        return nn.Dense(self.dim, name="net_2")(h)


class CogVideoXBlock(nn.Module):
    dim: int
    num_heads: int
    ff_inner_dim: int
    norm_eps: float = 1e-5
    mesh: Any = None
    sequence_parallel: bool = False
    sp_axis_name: str = "sp"

    @nn.compact
    def __call__(self, hidden_states, encoder_hidden_states, temb, rope_cos, rope_sin):
        text_len = encoder_hidden_states.shape[1]

        norm_h, norm_e, gate_msa, enc_gate_msa = CogVideoXLayerNormZero(
            self.dim, self.norm_eps, name="norm1")(hidden_states, encoder_hidden_states, temb)
        attn_h, attn_e = CogVideoXAttention(
            self.dim, self.num_heads, mesh=self.mesh,
            sequence_parallel=self.sequence_parallel, sp_axis_name=self.sp_axis_name,
            name="attn1")(norm_h, norm_e, rope_cos, rope_sin)
        hidden_states = hidden_states + gate_msa * attn_h
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_e

        norm_h, norm_e, gate_ff, enc_gate_ff = CogVideoXLayerNormZero(
            self.dim, self.norm_eps, name="norm2")(hidden_states, encoder_hidden_states, temb)
        ff_out = CogVideoXFeedForward(self.dim, self.ff_inner_dim, name="ff")(
            jnp.concatenate([norm_e, norm_h], axis=1))
        hidden_states = hidden_states + gate_ff * ff_out[:, text_len:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_out[:, :text_len]
        return hidden_states, encoder_hidden_states


class CogVideoXPatchEmbed(nn.Module):
    """diffusers `CogVideoXPatchEmbed`. `text_proj` Linear; visual patchify
    is a per-frame `p x p` Conv2d (1.0) or a Linear over flattened
    `C*p*p*p_t` patches (1.5). Optional additive positional embedding is
    handled by the caller (`CogVideoXDiT`), not here.
    """
    dim: int
    in_channels: int
    patch_size: int = PATCH_SIZE
    patch_size_t: Optional[int] = None
    patch_bias: bool = True

    @nn.compact
    def __call__(self, text_embeds, image_embeds):
        # text_embeds: (B, 226, 4096); image_embeds: (B, F, C, H, W)
        b, f, c, h, w = image_embeds.shape
        p = self.patch_size
        text_embeds = nn.Dense(self.dim, name="text_proj")(text_embeds)

        if self.patch_size_t is None:
            x = jnp.transpose(image_embeds, (0, 1, 3, 4, 2)).reshape(b * f, h, w, c)  # (B*F, H, W, C)
            x = nn.Conv(self.dim, (p, p), strides=(p, p), padding="VALID",
                        use_bias=self.patch_bias, name="proj")(x)  # (B*F, H/p, W/p, dim)
            x = x.reshape(b, f, (h // p) * (w // p), self.dim).reshape(b, f * (h // p) * (w // p), self.dim)
        else:
            pt = self.patch_size_t
            x = jnp.transpose(image_embeds, (0, 1, 3, 4, 2))  # (B, F, H, W, C)
            x = x.reshape(b, f // pt, pt, h // p, p, w // p, p, c)
            # diffusers: permute(0,1,3,5,7,2,4,6) -> (B, F/pt, H/p, W/p, C, pt, p, p),
            # then the feature vector is flattened C-slowest as C*pt*p*p.
            x = jnp.transpose(x, (0, 1, 3, 5, 7, 2, 4, 6))
            x = x.reshape(b, (f // pt) * (h // p) * (w // p), c * pt * p * p)
            x = nn.Dense(self.dim, name="proj")(x)

        return jnp.concatenate([text_embeds, x], axis=1)  # (B, 226+Nv, dim)


class CogVideoXDiT(nn.Module):
    """CogVideoX's `CogVideoXTransformer3DModel`. Pass a preset from
    `vidax.models.cogvideo.configs.dit_kwargs(variant)` as constructor kwargs.

    `setup()` + `pre_process` / block-loop / `post_process` split mirrors
    `vidax.models.ltx_video.dit.LTXDiT` so `--offload_dit_weights` can stream
    one block's params into HBM at a time; `__call__` chains all three.
    """
    num_layers: int = 42
    num_attention_heads: int = 48
    in_channels: int = 16
    out_channels: int = 16
    use_rotary_positional_embeddings: bool = True
    use_learned_positional_embeddings: bool = False
    patch_size_t: Optional[int] = None
    patch_bias: bool = True
    ofs_embed_dim: Optional[int] = None
    sample_width: int = 90
    sample_height: int = 60
    sample_frames: int = 49
    spatial_interpolation_scale: float = 1.875
    temporal_interpolation_scale: float = 1.0
    max_text_seq_length: int = TEXT_SEQ_LENGTH
    norm_eps: float = 1e-5
    compute_dtype: jnp.dtype = jnp.bfloat16
    mesh: Any = None  # Megatron TP mesh; when set, attention uses the TPU flash kernel.
    # DeepSpeed-Ulysses sequence parallelism: shards the visual token sequence
    # across the mesh's 'sp' axis between blocks (text prefix stays replicated),
    # reshuffling to a head-sharded full-sequence view only for the joint
    # attention itself. Needed for CogVideoX-1.5 at its native 1360x768 (~45k
    # visual tokens), whose per-block activations otherwise don't fit a v4 chip.
    # Requires the whole `apply(...)` to run inside `shard_map(..., mesh=mesh)` --
    # see `examples/generate_cogvideox.py`. Composes with `mesh` (Megatron TP)
    # only trivially here: CogVideoX keeps TP and SP mutually exclusive (the
    # example asserts `tensor_parallel_size == 1` whenever `sequence_parallel`),
    # so no column/row-parallel shape juggling is threaded through this module.
    sequence_parallel: bool = False
    sp_axis_name: str = "sp"

    def setup(self):
        if self.sequence_parallel:
            # The SP path chunks only the visual tokens and never touches the
            # learned-buffer / 3D-sincos positional paths (which would also need
            # chunking). CogVideoX-1.5 -- the only variant that needs SP -- is
            # rotary, so this is a real restriction only for 2b / 5b-I2V, which
            # fit at their native resolution without it.
            assert self.use_rotary_positional_embeddings and not self.use_learned_positional_embeddings, (
                "sequence_parallel is only supported for the rotary-positional-embedding "
                "CogVideoX variants (5b / 1.5-5b / 1.5-5b-i2v)")
        self.inner_dim = self.num_attention_heads * ATTENTION_HEAD_DIM
        self.patch_embed = CogVideoXPatchEmbed(
            self.inner_dim, self.in_channels, PATCH_SIZE, self.patch_size_t, self.patch_bias,
            name="patch_embed")
        self.time_embedding_linear_1 = nn.Dense(TIME_EMBED_DIM, name="time_embedding_linear_1")
        self.time_embedding_linear_2 = nn.Dense(TIME_EMBED_DIM, name="time_embedding_linear_2")
        if self.ofs_embed_dim is not None:
            self.ofs_embedding_linear_1 = nn.Dense(self.ofs_embed_dim, name="ofs_embedding_linear_1")
            self.ofs_embedding_linear_2 = nn.Dense(self.ofs_embed_dim, name="ofs_embedding_linear_2")
        if self.use_learned_positional_embeddings:
            # persistent buffer in the checkpoint (5b-I2V); shape
            # (1, max_text_seq_length + num_patches, inner_dim).
            post_h = self.sample_height // PATCH_SIZE
            post_w = self.sample_width // PATCH_SIZE
            post_t = (self.sample_frames - 1) // 4 + 1
            self.pos_embedding = self.param(
                "pos_embedding", nn.initializers.zeros,
                (1, self.max_text_seq_length + post_t * post_h * post_w, self.inner_dim))

        ff_inner_dim = self.inner_dim * 4
        self.blocks = [
            CogVideoXBlock(self.inner_dim, self.num_attention_heads, ff_inner_dim, self.norm_eps,
                           mesh=self.mesh, sequence_parallel=self.sequence_parallel,
                           sp_axis_name=self.sp_axis_name, name=f"transformer_blocks_{i}")
            for i in range(self.num_layers)
        ]
        self.norm_final_scale = self.param("norm_final_scale", nn.initializers.ones, (self.inner_dim,))
        self.norm_final_bias = self.param("norm_final_bias", nn.initializers.zeros, (self.inner_dim,))
        self.norm_out_linear = nn.Dense(2 * self.inner_dim, name="norm_out_linear")
        self.norm_out_norm_scale = self.param("norm_out_norm_scale", nn.initializers.ones, (self.inner_dim,))
        self.norm_out_norm_bias = self.param("norm_out_norm_bias", nn.initializers.zeros, (self.inner_dim,))
        p = PATCH_SIZE
        out_dim = p * p * self.out_channels if self.patch_size_t is None else \
            self.patch_size_t * p * p * self.out_channels
        self.proj_out = nn.Dense(out_dim, name="proj_out")

    def _time_embed(self, timestep, ofs, dtype):
        t_sin = _timestep_sinusoidal_embedding(timestep, self.inner_dim).astype(dtype)
        emb = self.time_embedding_linear_2(nn.silu(self.time_embedding_linear_1(t_sin)))
        if self.ofs_embed_dim is not None and ofs is not None:
            o_sin = _timestep_sinusoidal_embedding(ofs, self.ofs_embed_dim).astype(dtype)
            emb = emb + self.ofs_embedding_linear_2(nn.silu(self.ofs_embedding_linear_1(o_sin)))
        return emb

    def pre_process(self, hidden_states, encoder_hidden_states, timestep, rope_cos, rope_sin, ofs=None):
        dtype = self.compute_dtype
        emb = self._time_embed(timestep, ofs, dtype)

        x = self.patch_embed(encoder_hidden_states.astype(dtype), hidden_states.astype(dtype))
        if self.use_learned_positional_embeddings:
            x = x + self.pos_embedding[:, : x.shape[1]].astype(dtype)
        elif not self.use_rotary_positional_embeddings:
            # CogVideoX-2b: a fixed 3D sincos positional embedding over the
            # visual tokens (recomputed, not a checkpoint buffer -- matches
            # diffusers `CogVideoXPatchEmbed._get_positional_embeddings`,
            # embed_dim = inner_dim, grid = post-patch latent grid).
            _, f, _, h, w = hidden_states.shape
            pw, ph = w // PATCH_SIZE, h // PATCH_SIZE
            sincos = get_3d_sincos_pos_embed(
                self.inner_dim, (pw, ph), f,
                self.spatial_interpolation_scale, self.temporal_interpolation_scale)
            x = x.at[:, self.max_text_seq_length:].add(jnp.asarray(sincos, dtype))

        text_len = encoder_hidden_states.shape[1]
        enc = x[:, :text_len]
        vis = x[:, text_len:]
        rc = None if rope_cos is None else rope_cos.astype(dtype)
        rs = None if rope_sin is None else rope_sin.astype(dtype)

        if self.sequence_parallel:
            # Shard the visual token sequence across the 'sp' mesh axis -- every
            # block's per-token activations (norms, modulation, FFN) then only
            # cover this device's 1/sp_size chunk. The text prefix (`enc`) and
            # per-sample modulation (`emb`) stay replicated; the RoPE tables are
            # position-indexed identically to the visual tokens, so they're
            # chunked the same way. `CogVideoXAttention` reshuffles to a
            # full-sequence view internally for the joint attention itself.
            sp_size = self.mesh.shape[self.sp_axis_name]
            nv = vis.shape[1]
            assert nv % sp_size == 0, (
                f"sequence_parallel requires the visual token count ({nv}) to be "
                f"evenly divisible by the sequence-parallel size ({sp_size})")
            rank = jax.lax.axis_index(self.sp_axis_name)
            vis = chunk_by_rank(vis, 1, sp_size, rank)
            if rc is not None:
                rc = chunk_by_rank(rc, 0, sp_size, rank)
                rs = chunk_by_rank(rs, 0, sp_size, rank)
        return vis, enc, emb, rc, rs

    def post_process(self, vis, emb):
        vis = _layernorm(vis, self.norm_final_scale, self.norm_final_bias, self.norm_eps)
        mod = self.norm_out_linear(nn.silu(emb))
        shift, scale = jnp.split(mod, 2, axis=-1)
        vis = _layernorm(vis, self.norm_out_norm_scale, self.norm_out_norm_bias, self.norm_eps)
        vis = vis * (1 + scale[:, None, :]) + shift[:, None, :]
        return self.proj_out(vis)

    def __call__(self, hidden_states, encoder_hidden_states, timestep, rope_cos=None, rope_sin=None, ofs=None):
        """
        Args:
            hidden_states: (B, F, C, H, W) video latent (C == in_channels;
                for I2V the conditioning-image latent is already concatenated
                onto C by the caller).
            encoder_hidden_states: (B, 226, 4096) T5 text embeddings.
            timestep: (B,) diffusion timestep in [0, 1000].
            rope_cos, rope_sin: (Nv, 64) RoPE tables (visual tokens), or None
                for the non-rotary 2b checkpoint.
            ofs: (B,) offset value for 1.5-5B-I2V, else None.

        Returns:
            (B, F, out_channels, H, W) velocity prediction (patch dims folded
            back in; for 1.5 the F axis is F // patch_size_t before this and
            expanded here).
        """
        b, f, c, h, w = hidden_states.shape
        vis, enc, emb, rc, rs = self.pre_process(
            hidden_states, encoder_hidden_states, timestep, rope_cos, rope_sin, ofs)
        for block in self.blocks:
            vis, enc = block(vis, enc, emb, rc, rs)
        out = self.post_process(vis, emb)  # (B, Nv[/sp], out_dim)

        if self.sequence_parallel:
            # Re-assemble the full visual token sequence (each device's local
            # chunk, in rank order) before unpatchify.
            out = jax.lax.all_gather(out, self.sp_axis_name, axis=1, tiled=True)

        p = PATCH_SIZE
        if self.patch_size_t is None:
            out = out.reshape(b, f, h // p, w // p, self.out_channels, p, p)
            out = jnp.transpose(out, (0, 1, 4, 2, 5, 3, 6))
            out = out.reshape(b, f, self.out_channels, h, w)
        else:
            pt = self.patch_size_t
            out = out.reshape(b, f // pt, h // p, w // p, self.out_channels, pt, p, p)
            out = jnp.transpose(out, (0, 1, 5, 4, 2, 6, 3, 7))
            out = out.reshape(b, f, self.out_channels, h, w)
        return out
