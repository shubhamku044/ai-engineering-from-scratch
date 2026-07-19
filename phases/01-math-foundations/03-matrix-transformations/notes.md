# Phase 1 · Lesson 03 — Matrix Transformations · notes

> Built in scaffold mode (mentor wrote blueprints + tests + hints; I wrote every
> implementation body). Drafted from how I explained it. Mine to revise.

## What I built
`TransformMatrix(OperableMatrix)` — extends lesson 02, so it inherits `@`, `transpose`,
`determinant`, `inverse` for free:
- transforms as classmethods: `rotation`, `scaling`, `reflection_x`, `shear_x`
  (the columns of each matrix = where `[1,0]` and `[0,1]` land)
- `eigenvalues()` — from the characteristic equation `λ² − trace·λ + det = 0`, **reusing
  the inherited `determinant()`** instead of recomputing (the payoff of extending)
- `eigenvector_for(λ)` — solve `(A − λI)v = 0`, normalized

## Composition — order matters
Composing transforms = multiplying their matrices; "apply B then A" = `(A @ B) @ v`.
`(S@R)@v` gave `[~0, 1]` and `(R@S)@v` gave `[~0, 2]` — different, so matrix multiply is
non-commutative. (The `~0` was `cos(π/2)=1.2e-16` in floating point — compare with a
tolerance, never `==`.)

## Eigenvalues intuition (why they matter)
- An eigenvector is the direction a matrix only **stretches**, never rotates; λ = how much.
- **PCA:** eigenvectors of the covariance matrix are the directions the data spreads along;
  a large eigenvalue = data varies a lot there = most information. PCA keeps the top ones.
- **RNN stability:** applying `W` each timestep multiplies an eigen-component by λ every
  step, so `λⁿ`. `|λ|<1` → fades to 0 (vanishing); `|λ|>1` → explodes. That's the
  vanishing/exploding gradient problem, and why gating + gradient clipping exist.

## Design note
Extends `OperableMatrix` so knowledge compounds. Caveat: `@` still returns a base `Matrix`
(inherited, not overridden), so a composed product isn't a `TransformMatrix` — fine here,
since eigenvalue methods run on the originals.

## Definition of done (protocol check)
- [x] Wrote every implementation body myself (scaffold mode, no copying)
- [x] Can explain eigenvalues → PCA and → RNN stability out loud
- [x] All tests pass
