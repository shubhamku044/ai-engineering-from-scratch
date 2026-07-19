# Phase 1 · Lesson 04 — Calculus for ML · notes

> Scaffold mode (mentor wrote blueprints + tests; I wrote every body). Drafted
> from how I explained it. Mine to revise.

## What I built
- `numerical_derivative` — central difference `(f(x+h) − f(x−h)) / 2h` (more accurate
  than forward difference for the same h).
- `sigmoid` and `sigmoid_derivative` = `σ(x)(1 − σ(x))` (derived by hand; test checks it
  against the numerical derivative).
- `gradient` — numerical partial derivative per coordinate (nudge one axis, hold the rest).
- `gradient_descent` — `point ← point − lr · grad(point)`, repeated. The whole training loop.
- `fit_linear_regression` — trained `w, b` from scratch on `y = 2x + 1` data → recovered
  `w=2.0000, b=1.0000`. A model that **learned** the line, not one handed a formula.

## Gradient descent intuition
- **Learning rate:** too big → steps overshoot the minimum and bounce/diverge; too small →
  crawls, may not arrive in the budget of steps.
- **Why minus:** the gradient points uphill (steepest increase); we step the opposite way to
  reduce loss. `+` is gradient **ascent** — used when maximizing (e.g. a reward).
- **Local minima:** GD only follows the local downhill, so it can settle in a valley with no
  downhill exit. In million-weight nets this rarely hurts: most critical points are **saddle
  points** (an escape direction exists), and many minima have similarly low loss.

## The link forward
`dL/dw` and `dL/db` for the MSE loss were a chain-rule warm-up. Lesson 05 (autodiff/backprop)
generalizes exactly this — computing gradients through arbitrary compositions automatically.

## Definition of done (protocol check)
- [x] Wrote every body myself (scaffold mode, no copying)
- [x] Trained linear regression by gradient descent; can explain lr / negation / local minima
- [x] All tests pass
