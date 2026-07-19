# Phase 1 · Lesson 08 — Optimization · notes

> Scaffold mode. Drafted from how I explained it. Mine to revise.

## What I built
- `SGD` — vanilla (`momentum=0`) and with momentum (`v ← μv + g; p ← p − lr·v`).
- `Adam` — per-parameter adaptive steps from the 1st/2nd gradient moments with bias
  correction: `m,v` EMAs of `g` and `g²`, `m̂,v̂` correct the zero-init, step `= lr·m̂/(√v̂+ε)`.
- `minimize` — the generic train loop: `params = opt.step(params, grad_fn(params))` repeated.

## Momentum shines on ILL-CONDITIONED valleys (not round bowls)
On a symmetric bowl, momentum=0.9 overshoots and oscillates — it can end up *worse* than
plain GD. Its real benefit is elongated valleys (steep one way, shallow another): plain GD's
lr is capped by the steep axis so the shallow axis crawls; momentum builds velocity along the
slow axis and accelerates it. (My mentor's first test wrongly used a round bowl — fixed to
`f = x² + 25y²`, where momentum genuinely wins.)

## What Adam's /√v̂ buys
`v̂` is each weight's typical squared-gradient. Dividing the step by `√v̂` shrinks steps for
weights with big gradients and grows them for weights with tiny gradients — a per-parameter
learning rate. That's why one global `lr` handles Rosenbrock (steep in x, flat in y) and why
Adam needs far less lr tuning. Proof: Adam reached ≈(1,1) on Rosenbrock.

## Definition of done (protocol check)
- [x] Wrote SGD/Adam/minimize myself (scaffold mode)
- [x] Can explain momentum (when it helps) and Adam's per-parameter rescaling
- [x] All tests pass (incl. Adam on Rosenbrock)
