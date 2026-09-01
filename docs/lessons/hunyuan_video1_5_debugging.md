# HunyuanVideo-1.5 Debugging Lessons

Findings from porting HunyuanVideo-1.5's DiT/VAE/text-and-vision-encoders
to JAX/Flax. See
[`docs/models/hunyuan_video1_5.md`](../models/hunyuan_video1_5.md) for
the full port and its architecture.

## `attn_mode="flash"` masks only key positions, not query+key

The reference's real flash-attention path (`attn_mode="flash"`, what every
released checkpoint ships with) builds its mask as a single 1D per-*key*
validity vector, not a symmetric `(mask_q & mask_k)` 2D mask like its
`torch`-SDPA fallback branch — under variable-length packing, invalid keys
are excluded from every query's attention, but padded/invalid query rows
still get *some* computed output (from the valid keys). This matters
because a padded token's output still gets consumed as a *key* by later
layers, so a key-only-masked and symmetric-masked implementation could in
principle diverge — except every layer re-masks the same positions as
keys, so an invalid position's exact value never actually reaches a valid
query's output either way. Verified bit-exact (`corr=0.99999999998`) with a
key-only additive-bias implementation against the real PyTorch blocks,
including through the token-refiner's own masking — confirming the
key-only form is a correct match for the flash path, not just a
convenient shortcut.

## Flash attention with a dense bias still OOMs — use `SegmentIds`

`vidax.core.attention`'s bias-based masking path still materializes the
full `(B, H, S_q, S_k)` tensor before the Pallas kernel runs, defeating
flash attention's O(S) memory point entirely — invisible at small
synthetic sequence lengths, but a real `RESOURCE_EXHAUSTED` at actual video
resolution (one block's bias tensor alone requested 27.77G with 13.47G
free). Because this port's mask is always exactly "the first N key
positions are valid, the rest are padding" by construction (image tokens —
always valid — come first, text/glyph/vision padding is sorted to the
end), it maps directly onto Pallas's `SegmentIds` mechanism: give queries
and valid keys segment 1, invalid keys segment 0, and the kernel skips
mismatched blocks without ever materializing a dense tensor. General
lesson for any masked-attention Video DiT with a similarly structured
mask: prefer `SegmentIds` over an additive bias whenever the valid/invalid
split can be expressed as a segment partition — the memory difference is
existential, not marginal, at real sequence lengths.

## Tensor parallelism: every Pallas call in a partitioned program needs its own `shard_map`

Once *any* part of a `jax.jit`-compiled program is multi-device
GSPMD-partitioned (true the moment any param carries TP sharding), *every*
Pallas/Mosaic call anywhere in that same program needs a `shard_map`
wrapper — not just the calls whose own operands are physically split.
Missed this for `SingleTokenRefiner`'s attention (its weights are
deliberately left un-sharded — a fused QKV Dense's column-split doesn't
align with per-head boundaries), which crashed with `NotImplementedError:
Mosaic kernels cannot be automatically partitioned` despite that call's
own operands being fully replicated. Fixed with a second, fully-replicated
`shard_map` wrapper for the same kernel. See
[`docs/hardware_and_sharding.md`](../hardware_and_sharding.md) for the
general write-up of this lesson.
