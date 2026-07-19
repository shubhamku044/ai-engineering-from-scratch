# Phase 1 · Lesson 04 — Calculus for ML
# Derivatives tell you which way is downhill. Gradient descent walks down.
# Scaffold mode: fill every `raise NotImplementedError`; the tests are the spec.
import math
from collections.abc import Callable

# ── PART 1: derivatives ──────────────────────────────────────────────────────


def numerical_derivative(
    f: Callable[[float], float], x: float, h: float = 1e-4
) -> float:
    """Approximate f'(x) with the CENTRAL difference:  (f(x+h) - f(x-h)) / (2h).

    Central (not forward) because it's far more accurate for the same h.
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def sigmoid(x: float) -> float:
    """σ(x) = 1 / (1 + e^-x)."""
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x: float) -> float:
    """Analytical derivative of sigmoid: σ(x)·(1 - σ(x)).

    (Derive it once by hand so you know WHY it's this, not just that it is.)
    """
    s = sigmoid(x)
    return s * (1 - s)


# ── PART 2: the gradient (vector of partial derivatives) ─────────────────────


def gradient(
    f: Callable[[list[float]], float], point: list[float], h: float = 1e-4
) -> list[float]:
    """Numerical gradient: the partial derivative w.r.t. each coordinate.

    For each index i: nudge ONLY coordinate i by ±h (hold the rest fixed) and take
    the central difference. Returns a list the same length as `point`.
    """
    # Hint: loop i over range(len(point)); build point_plus / point_minus as copies
    # with point[i] ± h; partial_i = (f(point_plus) - f(point_minus)) / (2h).
    grad: list[float] = []
    for i in range(len(point)):
        point_plus = point.copy()
        point_minus = point.copy()

        point_plus[i] += h
        point_minus[i] -= h

        partial_i = (f(point_plus) - f(point_minus)) / (2 * h)
        grad.append(partial_i)
    return grad


# ── PART 3: gradient descent (THE algorithm — understand this cold) ───────────


def gradient_descent(
    grad_fn: Callable[[list[float]], list[float]],
    start: list[float],
    learning_rate: float,
    steps: int,
) -> list[float]:
    """Walk downhill: repeatedly step OPPOSITE the gradient.

    Update rule, applied `steps` times to every coordinate:
        point ← point − learning_rate · grad_fn(point)
    Return the final point. (This is the entire training loop of every model.)
    """
    point = start.copy()
    for _ in range(steps):
        grad = grad_fn(point)
        point = [p - learning_rate * g for p, g in zip(point, grad, strict=True)]
    return point


# ── PART 4: the payoff — linear regression trained by gradient descent ───────


def fit_linear_regression(
    xs: list[float],
    ys: list[float],
    learning_rate: float = 0.01,
    epochs: int = 5000,
) -> tuple[float, float]:
    """Fit y ≈ w·x + b by minimizing MSE with gradient descent. Return (w, b).

    Loss L = mean((w·x + b − y)²). Its gradients (derive these — chain rule preview):
        dL/dw = (2/n) · Σ (w·xᵢ + b − yᵢ) · xᵢ
        dL/db = (2/n) · Σ (w·xᵢ + b − yᵢ)
    Start w=0, b=0; each epoch, compute both gradients over all points and update
    w and b with the descent rule.
    """
    w = 0.0
    b = 0.0
    n = len(xs)

    for _ in range(epochs):
        dw = 0.0
        db = 0.0

        for x, y in zip(xs, ys, strict=True):
            error = w * x + b - y
            dw += error * x
            db += error
        dw = (2 / n) * dw
        db = (2 / n) * db

        w -= learning_rate * dw
        b -= learning_rate * db
    return w, b


# ── tests (the spec — make them all pass) ────────────────────────────────────

if __name__ == "__main__":
    TOL = 1e-3

    print("=== Numerical derivative ===")
    # d/dx x² = 2x → at x=3, slope is 6
    assert math.isclose(numerical_derivative(lambda x: x**2, 3.0), 6.0, abs_tol=TOL)
    # d/dx sin = cos → at 0, slope is 1
    assert math.isclose(numerical_derivative(math.sin, 0.0), 1.0, abs_tol=TOL)
    print("✓ numerical derivative matches known slopes")

    print("\n=== Sigmoid + its derivative ===")
    assert math.isclose(sigmoid(0.0), 0.5, abs_tol=TOL)
    assert math.isclose(sigmoid_derivative(0.0), 0.25, abs_tol=TOL)  # 0.5·(1−0.5)
    # analytical derivative should match the numerical one everywhere
    for x in (-2.0, -0.5, 1.0, 3.0):
        assert math.isclose(
            sigmoid_derivative(x), numerical_derivative(sigmoid, x), abs_tol=TOL
        )
    print("✓ analytical sigmoid' matches numerical")

    print("\n=== Gradient ===")
    # ∇(x² + y²) at (1,1) = [2, 2]
    g = gradient(lambda p: p[0] ** 2 + p[1] ** 2, [1.0, 1.0])
    assert math.isclose(g[0], 2.0, abs_tol=TOL) and math.isclose(g[1], 2.0, abs_tol=TOL)
    print("✓ gradient of x²+y² at (1,1) ≈ [2,2]")

    print("\n=== Gradient descent ===")
    # minimize x²+y², whose gradient is [2x, 2y]; from (5,5) it should reach ≈ (0,0)
    minimum = gradient_descent(lambda p: [2 * p[0], 2 * p[1]], [5.0, 5.0], 0.1, 200)
    assert math.isclose(minimum[0], 0.0, abs_tol=TOL) and math.isclose(
        minimum[1], 0.0, abs_tol=TOL
    )
    print(f"✓ descent reached {minimum} ≈ (0,0)")

    print("\n=== Linear regression (trained, not solved) ===")
    # data generated from y = 2x + 1 exactly → GD should recover w≈2, b≈1
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [2 * x + 1 for x in xs]
    w, b = fit_linear_regression(xs, ys, learning_rate=0.01, epochs=10000)
    print(f"learned: w={w:.4f}, b={b:.4f}")
    assert math.isclose(w, 2.0, abs_tol=1e-2) and math.isclose(b, 1.0, abs_tol=1e-2)
    print("✓ gradient descent recovered y = 2x + 1")

    print("\nAll tests passed! 🎉")
