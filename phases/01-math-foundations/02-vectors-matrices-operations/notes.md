# Phase 1 · Lesson 02 — Vectors, Matrices & Operations · notes

> Drafted from how I explained it while building. Mine to revise.

## What I built
Reused my lesson-01 `Vector`/`Matrix`, then `OperableMatrix(Matrix)` adding:
- element-wise `__add__`, scalar `*` (both sides via `__rmul__`), `hadamard`
- `identity(n)` (classmethod), `determinant()` (recursive Laplace/cofactor expansion
  on minors), `inverse()` (Gauss-Jordan, raises on singular)
- `relu` + `dense_layer(W, x, b) = relu(W @ x + b)` — one real neural-network layer.

## Element-wise vs matrix multiply (the shape-bug lesson)
- **Matrix multiply (`@`)** — e.g. a dense layer `W @ x`: `W` is `(out × in)`, `x` is
  `(in,)`. Each output neuron depends on **all** input features, so the operation mixes
  everything and changes dimension. Inner dims must match.
- **Element-wise (`hadamard`/`*`)** — e.g. a **dropout mask**: multiply activations by a
  same-shape vector of 0/1. Each neuron is kept or dropped **independently**; positions
  must not mix, so shape in = shape out. (Same story for LSTM/GRU gates: `gate * state`.)

## Design tradeoff I made (worth fixing later)
My new ops return a base `Matrix`, not `OperableMatrix`, so a result loses `.determinant()`
etc. — the subclass isn't "closed" under its own operations. Cleaner would be to return
`type(self)(...)` so chaining stays within the subclass.

## Definition of done (protocol check)
- [x] Built element-wise ops, determinant, inverse, dense layer from a blank file
- [x] Can explain when frameworks use element-wise vs matrix multiply, with examples
- [x] All operations self-tested with asserts, run green
