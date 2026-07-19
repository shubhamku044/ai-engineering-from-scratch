# Phase 1 · Lesson 06 — Probability & Distributions · notes

> Scaffold mode. Drafted from how I explained it. Mine to revise.

## What I built
- `normal_pdf`, `expected_value` (Σ xᵢpᵢ), `variance` (Σ pᵢ(xᵢ−μ)²).
- `softmax` — logits → a probability distribution (sums to 1).
- `log_softmax` — stable `log(softmax)`.
- `cross_entropy(logits, target)` = `−log_softmax(logits)[target]`.

## softmax numerical stability (the trick that matters)
Subtracting `max(logits)` from every logit subtracts the same constant from all of them,
which multiplies every `exp` by the same factor — and that factor cancels in the
numerator/denominator ratio, so the probabilities are unchanged. But now the largest term
is `exp(0)=1` instead of `exp(1000)=inf`, so no overflow. Free correctness.

## cross-entropy = negative log-likelihood
CE for one example is `−log(p_target)` where `p = softmax(logits)`. Minimizing it pushes the
model to assign high probability to the true class. Computed via `log_softmax` (not
`log(softmax(...))`) to avoid `log(0)`. This is *the* loss for classifiers and LLMs.

## Definition of done (protocol check)
- [x] Wrote every body myself (scaffold mode)
- [x] Can explain the softmax max-subtraction trick and CE = NLL
- [x] All tests pass
