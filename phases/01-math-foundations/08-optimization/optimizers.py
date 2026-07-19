# Phase 1 · Lesson 08 — Optimization
# Optimizers are strategies for walking downhill. You already wrote vanilla GD in
# lesson 04; here you add momentum and Adam — the two ideas behind how real models
# actually train. Same interface as PyTorch: opt.step(params, grads) -> new params.
# Scaffold mode: fill every `raise NotImplementedError`; the tests are the spec.
import math
from collections.abc import Callable

# ── optimizers (stateful, like torch.optim) ──────────────────────────────────


class SGD:
    """Stochastic gradient descent, optionally with momentum.

    momentum=0 → vanilla GD: param -= lr * grad
    momentum>0 → accumulate a velocity so consistent gradients build speed and
                 noisy ones cancel out:
                     v ← momentum * v + grad
                     param ← param − lr * v
    """

    def __init__(self, lr: float, momentum: float = 0.0) -> None:
        self.lr = lr
        self.momentum = momentum
        self.velocity: list[float] | None = None  # lazily sized to params

    def step(self, params: list[float], grads: list[float]) -> list[float]:
        if self.velocity is None:
            self.velocity = [0.0] * len(params)
        # For each i: self.velocity[i] = momentum*velocity[i] + grads[i]
        #             new_param_i     = params[i] - lr*velocity[i]
        # Return the new params list (and keep self.velocity updated for next step).
        new_params: list[float] = []

        for i in range(len(params)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grads[i]
            new_params.append(params[i] + self.velocity[i])
        return new_params


class Adam:
    """Adam: per-parameter adaptive learning rates from the 1st & 2nd moments of
    the gradient. This is the default optimizer for most deep learning.

    Per step (t counts from 1):
        m ← β1·m + (1−β1)·g            # 1st moment (mean of grads)
        v ← β2·v + (1−β2)·g²           # 2nd moment (mean of squared grads)
        m̂ ← m / (1 − β1^t)             # bias correction (m,v start at 0)
        v̂ ← v / (1 − β2^t)
        param ← param − lr · m̂ / (√v̂ + eps)
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: list[float] | None = None
        self.v: list[float] | None = None
        self.t = 0

    def step(self, params: list[float], grads: list[float]) -> list[float]:
        # Lazily size the moment buffers on the first step. Checking both m and v
        # also narrows them from `list[float] | None` to `list[float]` below.
        if self.m is None or self.v is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)
        self.t += 1

        new_params: list[float] = []

        for i in range(len(params)):
            g = grads[i]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            new_params.append(
                params[i] - self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
            )
        return new_params


# ── driver: run an optimizer to a minimum ────────────────────────────────────


def minimize(
    grad_fn: Callable[[list[float]], list[float]],
    start: list[float],
    optimizer: "SGD | Adam",
    steps: int,
) -> list[float]:
    """Generic training loop: repeatedly ask grad_fn for gradients and let the
    optimizer update the params. Return the final params."""
    # params = start.copy()
    # for _ in range(steps): params = optimizer.step(params, grad_fn(params))
    # return params
    params = start.copy()
    for _ in range(steps):
        grads = grad_fn(params)
        params = optimizer.step(params, grads)
    return params


# ── tests (the spec — make them all pass) ────────────────────────────────────

if __name__ == "__main__":
    TOL = 1e-2

    # Convex bowl f(x,y) = x² + y², gradient [2x, 2y], minimum at (0,0).
    def bowl_grad(p: list[float]) -> list[float]:
        return [2 * p[0], 2 * p[1]]

    print("=== Vanilla SGD ===")
    out = minimize(bowl_grad, [5.0, 5.0], SGD(lr=0.1), steps=200)
    assert all(math.isclose(c, 0.0, abs_tol=TOL) for c in out), out
    print(f"✓ SGD reached {out} ≈ (0,0)")

    print("\n=== SGD + momentum on an ILL-CONDITIONED valley ===")

    # f = x² + 25y²: steep in y, shallow in x. Plain GD's lr is capped by the steep
    # y-axis, so the shallow x-axis crawls. Momentum accumulates velocity along that
    # slow axis and accelerates it — the actual reason momentum exists. (On a round
    # bowl momentum just overshoots; that's why this test uses an elongated valley.)
    def valley_grad(p: list[float]) -> list[float]:
        return [2 * p[0], 50 * p[1]]

    def dist(p: list[float]) -> float:
        return math.hypot(p[0], p[1])

    plain = minimize(valley_grad, [5.0, 5.0], SGD(lr=0.02), steps=100)
    moment = minimize(valley_grad, [5.0, 5.0], SGD(lr=0.02, momentum=0.9), steps=100)
    assert dist(moment) < dist(plain), (
        f"momentum {dist(moment)} should beat plain {dist(plain)} on ill-conditioned valley"
    )
    print(f"✓ momentum dist={dist(moment):.4f} < plain dist={dist(plain):.4f}")

    print("\n=== Adam ===")
    out = minimize(bowl_grad, [5.0, 5.0], Adam(lr=0.1), steps=500)
    assert all(math.isclose(c, 0.0, abs_tol=TOL) for c in out), out
    print(f"✓ Adam reached {out} ≈ (0,0)")

    print("\n=== Adam on Rosenbrock (the hard, curved valley) ===")

    # f(x,y) = (1-x)² + 100(y-x²)²  → minimum at (1,1). Notoriously hard for plain GD.
    def rosen_grad(p: list[float]) -> list[float]:
        x, y = p
        return [-2 * (1 - x) - 400 * x * (y - x**2), 200 * (y - x**2)]

    out = minimize(rosen_grad, [-1.0, 1.0], Adam(lr=0.01), steps=20000)
    assert math.isclose(out[0], 1.0, abs_tol=0.1) and math.isclose(
        out[1], 1.0, abs_tol=0.1
    ), out
    print(f"✓ Adam navigated Rosenbrock to {out} ≈ (1,1)")

    print("\nAll tests passed! 🎉")
