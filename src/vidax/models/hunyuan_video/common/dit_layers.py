"""Shared DiT building blocks for HunyuanVideo / HunyuanVideo-1.5.

Structural port of the dual-stream + single-stream MMDiT blocks shared
(verbatim class names/shapes) by both reference repos:
  - ``hyvideo/models/transformers/hunyuanvideo_1_5_transformer.py``
    (``MMDoubleStreamBlock``, ``MMSingleStreamBlock``)
  - ``hyvideo/models/transformers/modules/{modulate_layers,mlp_layers,
    token_refiner,norm_layers}.py``

Only the ``attn_mode="flash"`` code path (what the real checkpoints ship
with -- see each checkpoint's ``config.json``, ``"attn_mode": "flash"``) is
implemented; the ``torch``/``flex-block-attn``/``sageattn`` fallback paths
in the reference are irrelevant here.

**Key-only attention masking.** The reference's joint (image+text)
attention receives a 1D per-key validity mask (``text_mask``, extended to
cover byT5/vision tokens -- see ``hunyuan_video_1_5/dit.py``), left-padded
with ``True`` for every image-token key position (``F.pad(text_mask,
(sequence_length, 0), value=True)`` in ``modules/attention.py``'s
``sequence_parallel_attention``). Under ``flash_attn_no_pad``'s
variable-length packing, this masks out invalid KEY positions everywhere
padded/invalid QUERY rows' own outputs are simply not read downstream (the
final image comes only from the image-token slice; padded text-token
outputs still get consumed as *keys* in later layers, where they remain
masked again). A dense JAX implementation that adds a large negative bias
to invalid key positions -- for every query row, including invalid ones --
is therefore provably equivalent at every position that ends up feeding the
final output, without needing to replicate flash-attn's query-side
unpadding. This is what ``joint_attention``/``masked_self_attention`` below
implement; do not "fix" it into a symmetric query+key mask (the reference's
own ``torch``-SDPA fallback branch does that and is *not* what the shipped
checkpoints were run with).
"""
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from vidax.core.attention import RMSNorm, _flash_attention_tpu, _pad_seq, _QKV_SPEC
from vidax.models.hunyuan_video.common.rope import apply_rope3d

_NEG_INF = -1e9


def modulate(x: jnp.ndarray, shift: Optional[jnp.ndarray], scale: Optional[jnp.ndarray]) -> jnp.ndarray:
    """``x * (1 + scale) + shift``, shift/scale broadcast over the sequence axis.

    Matches ``modules/modulate_layers.py:modulate`` exactly (shift/scale
    each shaped (B, C), unsqueezed to (B, 1, C)).
    """
    if scale is None and shift is None:
        return x
    if shift is None:
        return x * (1 + scale[:, None, :])
    if scale is None:
        return x + shift[:, None, :]
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


def apply_gate(x: jnp.ndarray, gate: Optional[jnp.ndarray]) -> jnp.ndarray:
    """``x * gate``, gate (B, C) broadcast over the sequence axis."""
    if gate is None:
        return x
    return x * gate[:, None, :]


class ModulateDiT(nn.Module):
    """Zero-init ``Linear(hidden, factor*hidden)`` after SiLU -- ``ModulateDiT``."""
    hidden_size: int
    factor: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.silu(x)
        return nn.Dense(
            self.factor * self.hidden_size,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="linear",
        )(x)


class MLP(nn.Module):
    """``fc1 -> act -> fc2``, matching ``modules/mlp_layers.py:MLP`` (no dropout at inference)."""
    hidden_channels: int
    out_features: int
    act: str = "gelu_tanh"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_channels, name="fc1")(x)
        x = _activation(self.act)(x)
        x = nn.Dense(self.out_features, name="fc2")(x)
        return x


def _activation(name: str):
    if name == "gelu_tanh":
        return lambda x: nn.gelu(x, approximate=True)
    if name == "gelu":
        return lambda x: nn.gelu(x, approximate=False)
    if name == "silu":
        return nn.silu
    if name == "relu":
        return nn.relu
    raise NotImplementedError(f"Unsupported activation: {name}")


def _flash_attention_tpu_segment_masked(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
    key_valid: jnp.ndarray, scale: float,
) -> jnp.ndarray:
    """Key-only-masked flash attention via Pallas ``SegmentIds``, not a
    dense additive bias.

    ``_flash_attention_tpu(..., bias=...)`` (``vidax.core.attention``)
    still *materializes* a full ``(B, H, S_q, S_k)`` bias tensor before
    calling the kernel (`jnp.broadcast_to(bias, ...)`) -- for HunyuanVideo-
    1.5's real joint image+text sequence lengths (tens of thousands of
    tokens) that alone OOMs a TPU v4 chip, defeating the point of using
    flash attention at all. Our mask is always exactly "the first N key
    positions are valid, the rest are padding" (`dit.py`'s
    `_reorder_tokens` guarantees this -- image tokens are always valid and
    come first, followed by reordered valid-then-invalid text/glyph/vision
    tokens), which is *exactly* the shape Pallas's own `SegmentIds`
    mechanism is built for: give every query position segment 1, give
    valid keys segment 1 and invalid keys segment 0, and the kernel skips
    whole mismatched blocks without ever materializing a dense (S, S)
    tensor -- true O(S) memory.
    """
    from jax.experimental.pallas.ops.tpu.flash_attention import SegmentIds, flash_attention

    b, sq, h, d = q.shape
    qt = jnp.transpose(q, (0, 2, 1, 3))
    kt = jnp.transpose(k, (0, 2, 1, 3))
    vt = jnp.transpose(v, (0, 2, 1, 3))

    qt, sq0 = _pad_seq(qt, axis=2)
    kt, sk0 = _pad_seq(kt, axis=2)
    vt, _ = _pad_seq(vt, axis=2)

    q_ids = jnp.ones((b, qt.shape[2]), dtype=jnp.int32)
    kv_valid = key_valid.astype(jnp.int32)
    kv_ids = jnp.pad(kv_valid, ((0, 0), (0, kt.shape[2] - kv_valid.shape[1])))
    segment_ids = SegmentIds(q=q_ids, kv=kv_ids)

    out = flash_attention(qt, kt, vt, segment_ids=segment_ids, sm_scale=scale)
    out = out[:, :, :sq0, :]
    return jnp.transpose(out, (0, 2, 1, 3))


def _flash_attention_tpu_segment_masked_sharded(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
    key_valid: jnp.ndarray, scale: float, mesh: Mesh,
) -> jnp.ndarray:
    """Tensor-parallel wrapper around `_flash_attention_tpu_segment_masked`.

    Pallas/Mosaic kernels are opaque custom calls GSPMD cannot
    auto-partition (same reason `vidax.core.attention.
    _flash_attention_tpu_sharded` needs `shard_map` at all) -- so whenever
    q/k/v are physically sharded across devices (Megatron TP: `num_heads`
    split across the mesh's `tp` axis, matching `vidax.core.sharding
    .shard_wan_params`'s column/row-parallel layout for this model's
    Q/K/V/output Dense layers), the flash-attention call must run inside an
    explicit `shard_map`, giving each device the kernel call over its own
    local heads with no cross-device communication needed (each device
    already owns a disjoint, complete subset of attention heads -- exactly
    the same reasoning `_flash_attention_tpu_sharded` documents). `key_valid`
    is replicated (every device needs the full mask; it doesn't depend on
    which heads a device owns), unlike `vidax.core.attention
    .dot_product_attention`'s dense-bias sharded path, which doesn't thread
    a mask through at all -- ours does, since `_flash_attention_tpu_segment_
    masked`'s `SegmentIds` mechanism costs nothing extra to carry through
    `shard_map` (still O(S), not O(S^2)).
    """
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    def _local(q, k, v, key_valid):
        return _flash_attention_tpu_segment_masked(q, k, v, key_valid, scale)

    key_valid_spec = P('dp', None)
    return shard_map(
        _local, mesh=mesh, in_specs=(_QKV_SPEC, _QKV_SPEC, _QKV_SPEC, key_valid_spec),
        out_specs=_QKV_SPEC, check_rep=False)(q, k, v, key_valid)


def _flash_attention_tpu_segment_masked_replicated(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
    key_valid: jnp.ndarray, scale: float, mesh: Mesh,
) -> jnp.ndarray:
    """Same Pallas-kernel-needs-`shard_map` reasoning as
    `_flash_attention_tpu_segment_masked_sharded`, but for a caller whose
    own weights/activations are **not** TP-sharded (`SingleTokenRefiner`'s
    fused `self_attn_qkv` -- deliberately left un-sharded, see
    `vidax.core.sharding`'s HunyuanVideo-1.5 whitelist comments, since a
    fused-QKV Dense's contiguous column split doesn't align with clean
    per-head boundaries).

    Once *any* part of a `jax.jit`-compiled program runs under multi-device
    GSPMD partitioning (true the moment `dit_params` carries any TP
    sharding), **every** Pallas/Mosaic call anywhere in that same program
    needs a `shard_map` wrapper -- not just the ones whose operands are
    actually physically split (confirmed by a real
    `NotImplementedError: Mosaic kernels cannot be automatically
    partitioned` crash inside `SingleTokenRefiner`'s attention when only
    the double/single-stream blocks' calls were wrapped). Every spec here
    is fully replicated (`P()`), so this call is a per-device no-op
    data-parallel-style replica of the same computation, matching how the
    surrounding (already-replicated) activations are actually laid out.
    """
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    def _local(q, k, v, key_valid):
        return _flash_attention_tpu_segment_masked(q, k, v, key_valid, scale)

    return shard_map(
        _local, mesh=mesh, in_specs=(P(), P(), P(), P()),
        out_specs=P(), check_rep=False)(q, k, v, key_valid)


def masked_self_attention(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
    key_valid: Optional[jnp.ndarray] = None,
    mesh: Optional[Mesh] = None,
    tp_sharded: bool = True,
) -> jnp.ndarray:
    """Multi-head self-attention with an optional key-only validity mask.

    Args:
        q, k, v: (B, S, H, D).
        key_valid: Optional (B, S) bool/int, True/1 where the key position
            is valid (see module docstring for why key-only, not a
            symmetric query+key mask, matches the reference). Must have
            all valid positions before all invalid positions (guaranteed
            by `dit.py`'s `_reorder_tokens` for every caller in this
            module) -- see `_flash_attention_tpu_segment_masked`.
        mesh: TP mesh, if this call may run inside a multi-device
            GSPMD-partitioned program (i.e. whenever *any* of the DiT's
            params carry TP sharding -- not just this specific call's own
            operands, see `tp_sharded`).
        tp_sharded: Whether *this caller's own* q/k/v are physically
            TP-sharded on the head axis (`MMDoubleStreamBlock`/
            `MMSingleStreamBlock`'s Q/K/V Dense layers are; `SingleTokenRefiner`'s
            fused `self_attn_qkv` deliberately isn't, see
            `vidax.core.sharding`). Selects which `shard_map` wrapper to
            use when `mesh` is given -- getting this wrong silently
            produces incorrect output (a head-sharded `shard_map` over a
            replicated array, or vice versa, both "work" without erroring
            but slice/replicate the wrong thing). Ignored when `mesh` is
            None.

    Returns:
        (B, S, H*D).

    On TPU this calls a Pallas flash-attention kernel (O(S) memory)
    instead of materializing the full (B, H, S, S) attention matrix --
    required at real video-token sequence lengths (tens of thousands of
    joint image+text tokens), where the naive dense einsum this function
    used during initial bit-exact verification (small synthetic S) OOMs a
    single TPU chip well before real resolutions. Not routed through
    `vidax.core.attention.dot_product_attention` itself: that function's
    multi-device/mesh dispatch heuristic checks *global*
    `jax.device_count()`, not whether this call's own q/k/v are actually
    sharded -- wrong for this port's current single-device-per-component
    placement (see `examples/generate_hunyuan_video_1_5.py`) on a
    multi-chip host; and its dense-bias flash path
    (`_flash_attention_tpu(..., bias=...)`) still materializes an O(S^2)
    tensor, unsuitable here regardless (see
    `_flash_attention_tpu_segment_masked`'s docstring). Revisit once
    sequence/tensor-parallel sharding lands for this model family (see
    docs/hardware_and_sharding.md).
    """
    b, s, h, d = q.shape
    scale = 1.0 / (d ** 0.5)

    if jax.devices()[0].platform == "tpu":
        if key_valid is not None:
            if mesh is not None and tp_sharded:
                out = _flash_attention_tpu_segment_masked_sharded(q, k, v, key_valid, scale, mesh)
            elif mesh is not None:
                out = _flash_attention_tpu_segment_masked_replicated(q, k, v, key_valid, scale, mesh)
            else:
                out = _flash_attention_tpu_segment_masked(q, k, v, key_valid, scale)
        else:
            out = _flash_attention_tpu(q, k, v, None, scale)
        return out.reshape(b, s, h * d)

    logits = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
    if key_valid is not None:
        bias = jnp.where(key_valid.astype(bool)[:, None, None, :], 0.0, _NEG_INF)
        logits = logits + bias
    weights = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
    out = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return out.reshape(b, s, h * d)


class QKVHead(nn.Module):
    """Q/K/V projection + optional per-head RMSNorm, shared by double/single blocks."""
    hidden_size: int
    heads_num: int
    qkv_bias: bool = True
    qk_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        head_dim = self.hidden_size // self.heads_num
        q = nn.Dense(self.hidden_size, use_bias=self.qkv_bias, name="attn_q")(x)
        k = nn.Dense(self.hidden_size, use_bias=self.qkv_bias, name="attn_k")(x)
        v = nn.Dense(self.hidden_size, use_bias=self.qkv_bias, name="attn_v")(x)
        b, s, _ = x.shape
        q = q.reshape(b, s, self.heads_num, head_dim)
        k = k.reshape(b, s, self.heads_num, head_dim)
        v = v.reshape(b, s, self.heads_num, head_dim)
        if self.qk_norm:
            q = RMSNorm(head_dim, name="attn_q_norm")(q)
            k = RMSNorm(head_dim, name="attn_k_norm")(k)
        return q, k, v


class MMDoubleStreamBlock(nn.Module):
    """Dual-stream MMDiT block -- port of ``MMDoubleStreamBlock``.

    Separate img/txt modulation + QKV + norms, joint self-attention over
    the concatenated [img; txt] sequence, separate MLPs.
    """
    hidden_size: int
    heads_num: int
    mlp_width_ratio: float = 4.0
    mlp_act_type: str = "gelu_tanh"
    qk_norm: bool = True
    qkv_bias: bool = True
    mesh: Optional[Mesh] = None

    @nn.compact
    def __call__(
        self,
        img: jnp.ndarray, txt: jnp.ndarray, vec: jnp.ndarray,
        freqs: Tuple[jnp.ndarray, jnp.ndarray],
        key_valid: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        mlp_hidden = int(self.hidden_size * self.mlp_width_ratio)

        img_mod = ModulateDiT(self.hidden_size, factor=6, name="img_mod")(vec)
        (img_mod1_shift, img_mod1_scale, img_mod1_gate,
         img_mod2_shift, img_mod2_scale, img_mod2_gate) = jnp.split(img_mod, 6, axis=-1)

        txt_mod = ModulateDiT(self.hidden_size, factor=6, name="txt_mod")(vec)
        (txt_mod1_shift, txt_mod1_scale, txt_mod1_gate,
         txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = jnp.split(txt_mod, 6, axis=-1)

        img_normed = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6, name="img_norm1")(img)
        img_normed = modulate(img_normed, img_mod1_shift, img_mod1_scale)
        img_q, img_k, img_v = QKVHead(self.hidden_size, self.heads_num, self.qkv_bias, self.qk_norm, name="img_attn")(img_normed)
        img_q = apply_rope3d(img_q, freqs)
        img_k = apply_rope3d(img_k, freqs)

        txt_normed = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6, name="txt_norm1")(txt)
        txt_normed = modulate(txt_normed, txt_mod1_shift, txt_mod1_scale)
        txt_q, txt_k, txt_v = QKVHead(self.hidden_size, self.heads_num, self.qkv_bias, self.qk_norm, name="txt_attn")(txt_normed)

        q = jnp.concatenate([img_q, txt_q], axis=1)
        k = jnp.concatenate([img_k, txt_k], axis=1)
        v = jnp.concatenate([img_v, txt_v], axis=1)
        attn = masked_self_attention(q, k, v, key_valid=key_valid, mesh=self.mesh)
        img_len = img.shape[1]
        img_attn, txt_attn = attn[:, :img_len], attn[:, img_len:]

        img = img + apply_gate(nn.Dense(self.hidden_size, use_bias=self.qkv_bias, name="img_attn_proj")(img_attn), img_mod1_gate)
        img_mlp_in = modulate(nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6, name="img_norm2")(img), img_mod2_shift, img_mod2_scale)
        img = img + apply_gate(MLP(mlp_hidden, self.hidden_size, self.mlp_act_type, name="img_mlp")(img_mlp_in), img_mod2_gate)

        txt = txt + apply_gate(nn.Dense(self.hidden_size, use_bias=self.qkv_bias, name="txt_attn_proj")(txt_attn), txt_mod1_gate)
        txt_mlp_in = modulate(nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6, name="txt_norm2")(txt), txt_mod2_shift, txt_mod2_scale)
        txt = txt + apply_gate(MLP(mlp_hidden, self.hidden_size, self.mlp_act_type, name="txt_mlp")(txt_mlp_in), txt_mod2_gate)

        return img, txt


class MMSingleStreamBlock(nn.Module):
    """Fused single-stream MMDiT block -- port of ``MMSingleStreamBlock``.

    Operates on the pre-concatenated [img; txt] sequence with a single
    fused QKV+MLP-in projection and a single fused attn-out+MLP-out
    projection (``linear2``), single AdaLN modulation.
    """
    hidden_size: int
    heads_num: int
    mlp_width_ratio: float = 4.0
    mlp_act_type: str = "gelu_tanh"
    qk_norm: bool = True
    mesh: Optional[Mesh] = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray, vec: jnp.ndarray, txt_len: int,
        freqs: Tuple[jnp.ndarray, jnp.ndarray],
        key_valid: jnp.ndarray,
    ) -> jnp.ndarray:
        mlp_hidden = int(self.hidden_size * self.mlp_width_ratio)
        head_dim = self.hidden_size // self.heads_num

        mod = ModulateDiT(self.hidden_size, factor=3, name="modulation")(vec)
        mod_shift, mod_scale, mod_gate = jnp.split(mod, 3, axis=-1)
        x_mod = modulate(nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6, name="pre_norm")(x), mod_shift, mod_scale)

        b, s, _ = x_mod.shape
        q = nn.Dense(self.hidden_size, name="linear1_q")(x_mod).reshape(b, s, self.heads_num, head_dim)
        k = nn.Dense(self.hidden_size, name="linear1_k")(x_mod).reshape(b, s, self.heads_num, head_dim)
        v = nn.Dense(self.hidden_size, name="linear1_v")(x_mod).reshape(b, s, self.heads_num, head_dim)
        mlp_in = nn.Dense(mlp_hidden, name="linear1_mlp")(x_mod)

        if self.qk_norm:
            q = RMSNorm(head_dim, name="q_norm")(q)
            k = RMSNorm(head_dim, name="k_norm")(k)

        img_q, txt_q = q[:, :-txt_len], q[:, -txt_len:]
        img_k, txt_k = k[:, :-txt_len], k[:, -txt_len:]
        img_q = apply_rope3d(img_q, freqs)
        img_k = apply_rope3d(img_k, freqs)
        q = jnp.concatenate([img_q, txt_q], axis=1)
        k = jnp.concatenate([img_k, txt_k], axis=1)

        attn = masked_self_attention(q, k, v, key_valid=key_valid, mesh=self.mesh)
        mlp_act = _activation(self.mlp_act_type)(mlp_in)
        fused_in = jnp.concatenate([attn, mlp_act], axis=-1)
        out = nn.Dense(self.hidden_size, name="linear2")(fused_in)

        return x + apply_gate(out, mod_gate)


class IndividualTokenRefinerBlock(nn.Module):
    """Self-attention + MLP block used inside ``SingleTokenRefiner``."""
    hidden_size: int
    heads_num: int
    mlp_width_ratio: float = 4.0
    mesh: Optional[Mesh] = None  # see `masked_self_attention`'s `tp_sharded` doc -- always False here.

    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray, key_valid: Optional[jnp.ndarray]) -> jnp.ndarray:
        head_dim = self.hidden_size // self.heads_num
        mlp_hidden = int(self.hidden_size * self.mlp_width_ratio)

        gates = nn.Dense(
            2 * self.hidden_size,
            kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros,
            name="adaLN_modulation_1",
        )(nn.silu(c))
        gate_msa, gate_mlp = jnp.split(gates, 2, axis=-1)

        normed = nn.LayerNorm(epsilon=1e-6, name="norm1")(x)
        qkv = nn.Dense(3 * self.hidden_size, name="self_attn_qkv")(normed)
        b, s, _ = normed.shape
        q, k, v = jnp.split(qkv.reshape(b, s, 3, self.heads_num, head_dim), 3, axis=2)
        q, k, v = q[:, :, 0], k[:, :, 0], v[:, :, 0]
        # No qk_norm here: IndividualTokenRefinerBlock is constructed with
        # the SingleTokenRefiner's default qk_norm=False (reference never
        # overrides it), so this is a plain (no RMSNorm) self-attention.
        attn = masked_self_attention(q, k, v, key_valid=key_valid, mesh=self.mesh, tp_sharded=False)
        x = x + apply_gate(nn.Dense(self.hidden_size, name="self_attn_proj")(attn), gate_msa)

        mlp_in = nn.LayerNorm(epsilon=1e-6, name="norm2")(x)
        mlp_out = MLP(mlp_hidden, self.hidden_size, act="silu", name="mlp")(mlp_in)
        x = x + apply_gate(mlp_out, gate_mlp)
        return x


class SingleTokenRefiner(nn.Module):
    """LLM-text-embedding refiner -- port of ``SingleTokenRefiner``.

    Projects raw text-encoder hidden states to ``hidden_size``, builds a
    timestep+masked-mean-pooled-text conditioning vector, and runs
    ``depth`` self-attention refiner blocks (masked-mean pooling and the
    attention mask both use ``mask`` if given, else unmasked mean/no mask).
    """
    in_channels: int
    hidden_size: int
    heads_num: int
    depth: int = 2
    mesh: Optional[Mesh] = None  # threaded into IndividualTokenRefinerBlock as tp_sharded=False.

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        t_emb = _timestep_embedder(self.hidden_size, name="t_embedder")(t)

        if mask is None:
            pooled = x.mean(axis=1)
        else:
            mask_f = mask.astype(jnp.float32)[..., None]
            pooled = (x * mask_f).sum(axis=1) / mask_f.sum(axis=1)
        c_emb = nn.Sequential([
            nn.Dense(self.hidden_size, name="c_embedder_linear_1"),
            nn.silu,
            nn.Dense(self.hidden_size, name="c_embedder_linear_2"),
        ])(pooled)
        c = t_emb + c_emb

        x = nn.Dense(self.hidden_size, name="input_embedder")(x)

        key_valid = mask
        if key_valid is not None:
            # Reference forces position 0 valid to avoid a fully-masked row;
            # harmless under key-only masking (see module docstring) but
            # replicated for exactness.
            key_valid = key_valid.at[:, 0].set(True)

        for i in range(self.depth):
            x = IndividualTokenRefinerBlock(self.hidden_size, self.heads_num, mesh=self.mesh, name=f"blocks_{i}")(x, c, key_valid)
        return x


def _timestep_embedder(hidden_size: int, frequency_embedding_size: int = 256, max_period: float = 10000.0, name: str = "t_embedder"):
    """Returns a small callable module: sinusoidal(256) -> Dense -> SiLU -> Dense.

    Matches ``embed_layers.py:TimestepEmbedder`` (``timestep_embedding`` is
    the same ``cat([cos, sin])`` formula as
    ``vidax.core.rope3d.sinusoidal_embedding_1d``, reused directly).
    """
    from vidax.core.rope3d import sinusoidal_embedding_1d

    class _TimestepEmbedder(nn.Module):
        @nn.compact
        def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
            freq = sinusoidal_embedding_1d(frequency_embedding_size, t)
            h = nn.Dense(hidden_size, name="mlp_0")(freq)
            h = nn.silu(h)
            h = nn.Dense(hidden_size, name="mlp_2")(h)
            return h

    return _TimestepEmbedder(name=name)


class FinalLayer(nn.Module):
    """AdaLN-modulated LayerNorm -> zero-init Linear -- port of ``FinalLayer``."""
    hidden_size: int
    out_dim: int  # patch_t * patch_h * patch_w * out_channels

    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        mod = nn.Dense(
            2 * self.hidden_size,
            kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros,
            name="adaLN_modulation_1",
        )(nn.silu(c))
        shift, scale = jnp.split(mod, 2, axis=-1)
        x = modulate(nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6, name="norm_final")(x), shift, scale)
        return nn.Dense(
            self.out_dim,
            kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros,
            name="linear",
        )(x)
