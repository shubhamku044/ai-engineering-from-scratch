# Phase 1 · Lesson 05 — Chain Rule & Automatic Differentiation
# A tiny reverse-mode autograd engine (micrograd-style). Each Value records how it
# was made; backward() replays the graph in reverse, multiplying local derivatives
# along the way (the chain rule) to fill in every .grad.
#
# Scaffold mode: the graph plumbing is given; YOU write the chain rule — the three
# `_backward` closures and the backward() pass. That's the whole idea of backprop.
import math
from collections.abc import Callable

# ── the autograd engine ──────────────────────────────────────────────────────


class Value:
    """A scalar that remembers the operation that produced it, so gradients can
    flow backward through the computation graph."""

    def __init__(
        self, data: float, _children: tuple["Value", ...] = (), _op: str = ""
    ) -> None:
        self.data = data
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: set[Value] = set(_children)
        self._op = _op

    def __add__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # ∂(a+b)/∂a = 1
            # ∂(a+b)/∂b = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # ∂(a*b)/∂a = b
            # ∂(a*b)/∂b = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            # d/dx tanh(x) = 1 - tanh²(x)
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Reverse-mode autodiff."""
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v: "Value") -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # dL/dL = 1
        self.grad = 1.0

        # Traverse graph backwards
        for node in reversed(topo):
            node._backward()

    # Convenience methods
    def __radd__(self, other: "Value | float") -> "Value":
        return self + other

    def __rmul__(self, other: "Value | float") -> "Value":
        return self * other

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: "Value | float") -> "Value":
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# ── tests (the spec — make them all pass) ────────────────────────────────────

if __name__ == "__main__":
    TOL = 1e-6

    print("=== Basic mul gradient ===")
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    # dc/da = b = 3 ; dc/db = a = 2
    assert math.isclose(a.grad, 3.0, abs_tol=TOL)
    assert math.isclose(b.grad, 2.0, abs_tol=TOL)
    print("✓ d(a*b): a.grad=3, b.grad=2")

    print("\n=== Gradient ACCUMULATION (the classic += bug catcher) ===")
    x = Value(3.0)
    y = x * x  # x used twice → dy/dx = 2x = 6 ONLY if grads accumulate
    y.backward()
    assert math.isclose(x.grad, 6.0, abs_tol=TOL), (
        f"expected 6, got {x.grad} (did you use += ?)"
    )
    print("✓ y=x*x → x.grad=6 (accumulation works)")

    print("\n=== A deeper graph ===")
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    e = a * b  # -6
    d = e + c  # 4
    f = Value(-2.0)
    loss = d * f  # -8
    loss.backward()
    # by hand: dloss/dd = f = -2 ; dloss/de = -2
    #          dloss/da = -2*b = 6 ; dloss/db = -2*a = -4
    assert math.isclose(a.grad, 6.0, abs_tol=TOL)
    assert math.isclose(b.grad, -4.0, abs_tol=TOL)
    print("✓ deeper graph grads match hand derivation")

    print("\n=== tanh + gradient check vs finite differences ===")

    def f_expr(v: Value) -> Value:
        return (v * Value(2.0) + Value(1.0)).tanh()  # tanh(2x + 1)

    xv = Value(0.5)
    out = f_expr(xv)
    out.backward()
    analytic = xv.grad
    # numerical central difference on the same expression (data only)
    h = 1e-6
    fp = f_expr(Value(0.5 + h)).data
    fm = f_expr(Value(0.5 - h)).data
    numeric = (fp - fm) / (2 * h)
    print(f"analytic={analytic:.6f}  numeric={numeric:.6f}")
    assert math.isclose(analytic, numeric, abs_tol=1e-4)
    print("✓ autodiff gradient matches numerical gradient check")

    print("\nAll tests passed! 🎉  You built backprop.")
