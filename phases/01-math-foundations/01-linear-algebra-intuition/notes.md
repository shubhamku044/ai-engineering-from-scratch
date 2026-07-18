# Phase 1 · Lesson 01 — Linear Algebra Intuition · notes

> Drafted from how I explained the geometry while building. Mine to revise.

## What I built (from scratch, no numpy)
- `Vector`: add/sub, `dot`, `magnitude`, `normalize`, `cosine_similarity`,
  `angle_between` (with a float clamp so `acos(1.0000000002)` can't blow up),
  plus dimension validation and zero-vector guards.
- `Matrix`: `@` for both `Matrix @ Vector` and `Matrix @ Matrix` (implemented by
  transposing the right operand to grab its columns), `transpose`, `shape`.
- `project`, `gram_schmidt`, `is_linearly_independent`.

## Dot product vs cosine similarity
Raw dot product mixes **magnitude** and **direction**. A frequent word like "the" can
have a larger embedding norm than "cat", so `the·the` can beat `cat·cat` just from length,
not meaning. Cosine similarity divides by the two magnitudes, leaving only how aligned the
directions are, in `[-1, 1]` — the honest similarity measure.

## Why a neural-network layer is "just" W @ x
Each **row** of `W` is a learned direction (a feature). Dotting a row with the input `x`
measures how strongly `x` aligns with that feature — geometrically it's the (scaled)
**projection** of `x` onto the weight vector. A big positive number = strong match;
near zero = little alignment. The whole layer = one such measurement per row.

## Rank → LoRA
LoRA assumes the weight **update** is **low-rank**: instead of a full 4096×4096 update
(~16.7M params) it learns two skinny matrices, 4096×16 and 16×4096, whose product
approximates the update (~131K params, ~128× fewer). Rank `r=16` caps how many independent
directions the update can move in — the base weights stay frozen, only the small factors train.

## Gram-Schmidt / independence trick
Gram-Schmidt subtracts each vector's projection onto every accepted basis vector, then
normalizes → orthonormal basis. If a vector collapses to ~0 after subtracting projections,
it was a combination of the others. So `is_linearly_independent` just runs Gram-Schmidt and
returns False if it raises on a collapse — reuse instead of a separate rank routine.

## Definition of done (protocol check)
- [x] Built Vector/Matrix/projection/Gram-Schmidt from a blank file, no copying
- [x] Can explain the geometry (dot=alignment, layer=projection, rank=LoRA) out loud
- [x] All operations self-tested with asserts, run green
