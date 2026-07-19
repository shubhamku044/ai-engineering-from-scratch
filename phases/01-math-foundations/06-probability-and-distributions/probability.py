# Phase 1 · Lesson 06 — Probability & Distributions
# Probability is how models express uncertainty. The ML-critical pieces: softmax
# (logits → a distribution) and cross-entropy (how wrong that distribution is).
# Scaffold mode: fill every `raise NotImplementedError`; the tests are the spec.
import math

# ── distributions & moments ──────────────────────────────────────────────────


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Gaussian probability density:  (1 / (σ√(2π))) · exp(−(x−μ)² / (2σ²))."""
    return (
        1
        / (sigma * math.sqrt(2 * math.pi))
        * math.exp(-((x - mu) ** 2) / (2 * sigma**2))
    )


def expected_value(values: list[float], probs: list[float]) -> float:
    """E[X] = Σ xᵢ · pᵢ  (the mean of a discrete distribution)."""
    return sum(x * p for x, p in zip(values, probs, strict=True))


def variance(values: list[float], probs: list[float]) -> float:
    """Var[X] = E[(X − μ)²] = Σ pᵢ · (xᵢ − μ)²,  where μ = E[X]."""
    # Reuse expected_value for μ.
    mu = expected_value(values, probs)
    return sum(p * (x - mu) ** 2 for x, p in zip(values, probs, strict=True))


# ── the two that power every classifier / LLM ────────────────────────────────


def softmax(logits: list[float]) -> list[float]:
    """Turn raw scores (logits) into a probability distribution (sums to 1).

    softmax(x)ᵢ = exp(xᵢ) / Σ exp(xⱼ)

    STABILITY TRICK (required): subtract max(logits) from every logit first.
    It doesn't change the result (exp ratio is identical) but stops exp() from
    overflowing to inf on large logits like 1000.
    """
    # m = max(logits); exps = [exp(x - m) for x in logits]
    # return [e / sum(exps) for e in exps]
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]


def log_softmax(logits: list[float]) -> list[float]:
    """log(softmax(x)), computed stably as:  (xᵢ − m) − log(Σ exp(xⱼ − m)),  m = max.

    Directly logging softmax output would risk log(0); this form is stable.
    """
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    sum_exps = sum(exps)
    return [(x - m) - math.log(sum_exps) for x in logits]


def cross_entropy(logits: list[float], target: int) -> float:
    """Cross-entropy loss for a single example given raw logits and the true class index

    CE = −log(softmax(logits)[target]) = −log_softmax(logits)[target].
    This is the negative log-likelihood of the correct class. Reuse log_softmax.
    """
    log_probs = log_softmax(logits)
    return -log_probs[target]


# ── tests (the spec — make them all pass) ────────────────────────────────────

if __name__ == "__main__":
    TOL = 1e-6

    print("=== normal_pdf ===")
    # peak of the standard normal at x=0 is 1/√(2π) ≈ 0.39894
    assert math.isclose(normal_pdf(0.0), 1 / math.sqrt(2 * math.pi), abs_tol=TOL)
    print("✓ normal_pdf(0) = 1/√(2π)")

    print("\n=== expectation & variance ===")
    faces: list[float] = [1, 2, 3, 4, 5, 6]
    fair = [1 / 6] * 6
    assert math.isclose(expected_value(faces, fair), 3.5, abs_tol=TOL)  # fair die mean
    assert math.isclose(
        variance(faces, fair), 35 / 12, abs_tol=TOL
    )  # known die variance
    print("✓ fair die: E[X]=3.5, Var=35/12")

    print("\n=== softmax ===")
    p = softmax([1.0, 2.0, 3.0])
    assert math.isclose(sum(p), 1.0, abs_tol=TOL)  # it's a distribution
    assert p[2] > p[1] > p[0]  # order preserved
    # uniform logits → uniform distribution
    assert all(math.isclose(q, 1 / 3, abs_tol=TOL) for q in softmax([5.0, 5.0, 5.0]))
    # stability: huge logits must NOT overflow
    big = softmax([1000.0, 1001.0, 1002.0])
    assert math.isclose(sum(big), 1.0, abs_tol=TOL) and all(
        math.isfinite(x) for x in big
    )
    print("✓ softmax sums to 1, order-preserving, overflow-safe")

    print("\n=== cross-entropy ===")
    # confident & correct → low loss; confident & wrong → high loss
    logits = [1.0, 2.0, 5.0]
    assert cross_entropy(logits, 2) < cross_entropy(logits, 0)
    # CE must equal −log_softmax at the target
    assert math.isclose(cross_entropy(logits, 2), -log_softmax(logits)[2], abs_tol=TOL)
    print("✓ cross-entropy = negative log-likelihood of the true class")

    print("\nAll tests passed! 🎉")
