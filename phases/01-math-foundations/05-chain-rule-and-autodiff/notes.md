# Phase 1 · Lesson 05 — Chain Rule & Autodiff · notes

> Scaffold mode (mentor gave the graph plumbing + tests; I wrote the chain rule).
> Drafted from how I explained it. Mine to revise.

## What I built
A micrograd-style `Value` class — reverse-mode autograd:
- each op (`+`, `*`, `tanh`) builds an output `Value` and stashes a `_backward` closure
  that applies the **local derivative × out.grad** and ACCUMULATES into inputs (`+=`)
- `backward()` — topological sort of the graph, seed the output `grad = 1.0` (dL/dL = 1),
  then walk nodes in **reverse** calling each `_backward()`
- validated with a gradient check: autodiff grad == numerical finite-difference grad.

## The chain rule per op
- `+` : local derivative 1 for both inputs → grad passes straight through.
- `*` : each input's local derivative is the *other* input's value.
- `tanh`: local derivative `1 − tanh²(x)`.

## Why reverse-mode wins
One forward pass caches every intermediate value; one backward pass reuses them to produce
the gradient for **every** weight at once (~2× a forward pass). Numerical differentiation
perturbs one weight at a time and reruns the whole forward pass — ~one forward per weight,
so a million weights ≈ a million forward passes. (Caching those intermediates is also why
training is memory-hungry.)

## Why gradients accumulate (+=)
A weight can reach the loss through many paths (a shared/embedding weight, or `x` in `x*x`).
The multivariate chain rule says the total gradient is the **sum** over all paths — each path
independently affects the loss. Overwrite loses paths; average shrinks the signal; sum is correct.

## Reverse order matters
A node's `.grad` must be fully accumulated before it pushes gradient to its inputs — reverse
topological order guarantees every consumer of a node runs before the node itself.

## Definition of done (protocol check)
- [x] Wrote the three `_backward` closures + backward() myself (scaffold mode)
- [x] Can explain reverse-mode efficiency + why grads sum over paths
- [x] Gradient check passes (autodiff == numerical)
