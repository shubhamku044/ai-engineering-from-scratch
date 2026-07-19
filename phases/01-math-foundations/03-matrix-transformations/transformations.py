# Phase 1 · Lesson 03 — Matrix Transformations
# A matrix is a machine that reshapes space. Build the transforms, compose them
# (order matters!), then crack eigenvalues/eigenvectors of a 2x2 by hand.
#
# Knowledge compounds: TransformMatrix EXTENDS OperableMatrix (lesson 02), so it
# inherits determinant(), inverse(), @, transpose() for free — the eigenvalue
# code reuses determinant() instead of recomputing it.
import math

from linear_algebra import Vector
from matrices import OperableMatrix


class TransformMatrix(OperableMatrix):
    """A 2x2 (or n×n) matrix with spatial transforms + eigen-decomposition,
    built on top of OperableMatrix (which is built on Matrix)."""

    # ── PART 1: transformation constructors ──────────────────────────────────
    # Each returns a TransformMatrix. The COLUMNS are where [1,0] and [0,1] land.

    @classmethod
    def rotation(cls, theta: float) -> "TransformMatrix":
        """Rotate counter-clockwise by `theta` radians: [[cos,-sin],[sin,cos]]."""
        # Hint: c, s = math.cos(theta), math.sin(theta); return cls([[c, -s], [s, c]])
        c = math.cos(theta)
        s = math.sin(theta)
        return cls([[c, -s], [s, c]])

    @classmethod
    def scaling(cls, sx: float, sy: float) -> "TransformMatrix":
        """
        Scale x by sx, y by sy: [[sx,0],[0,sy]] (off-diagonals 0 — no axis mixing).
        """
        return cls([[sx, 0], [0, sy]])

    @classmethod
    def reflection_x(cls) -> "TransformMatrix":
        """Reflect across the x-axis: (x,y)->(x,-y). Where do [1,0] and [0,1] go?"""
        return cls([[1, 0], [0, -1]])

    @classmethod
    def shear_x(cls, k: float) -> "TransformMatrix":
        """Horizontal shear: (x,y)->(x+k*y, y). [1,0] stays; [0,1] slides to [k,1]."""
        return cls([[1, k], [0, 1]])

    # ── PART 2: eigenvalues & eigenvectors (2x2, by hand) ────────────────────

    def eigenvalues(self) -> tuple[float, float]:
        """Real eigenvalues via the characteristic equation
        λ² - trace·λ + det = 0, REUSING inherited determinant().

        Raises ValueError if not 2x2, or if eigenvalues are complex (disc < 0).
        """
        # Step 1: guard  if self.shape != (2, 2): raise ValueError(...)
        # Step 2: trace = self.rows[0][0] + self.rows[1][1]
        # Step 3: det = self.determinant()          # <-- inherited from OperableMatrix!
        # Step 4: disc = trace**2 - 4*det ; if disc < 0: raise ValueError("complex")
        # Step 5: root = math.sqrt(disc) ; return ((trace+root)/2, (trace-root)/2)
        if self.shape != (2, 2):
            raise ValueError("eigenvalues only defined for 2x2 matrices")

        trace = self.rows[0][0] + self.rows[1][1]
        det = self.determinant()
        disc = trace**2 - 4 * det

        if disc < 0:
            raise ValueError("eigenvalues are complex")

        root = math.sqrt(disc)
        return ((trace + root) / 2, (trace - root) / 2)

    def eigenvector_for(self, eigenvalue: float) -> Vector:
        """A UNIT eigenvector for `eigenvalue`. Solve (A - λI) v = 0.

        Row 1 of (A - λI) is [a-λ, b], so (a-λ)x + b·y = 0. A clean family:
          • if b != 0:       v = [b, λ - a]      (plug in: (a-λ)b + b(λ-a) = 0 ✓)
          • elif c != 0:     v = [λ - d, c]      (from row 2: c·x + (d-λ)·y = 0)
          • else (diagonal): v = [1, 0] if λ == a else [0, 1]
        Then normalize().
        """
        if self.shape != (2, 2):
            raise ValueError("eigenvectors only defined for 2x2 matrices")
        a = self.rows[0][0]
        b = self.rows[0][1]
        c = self.rows[1][0]
        d = self.rows[1][1]

        if b != 0:
            v = Vector([b, eigenvalue - a])
        elif c != 0:
            v = Vector([eigenvalue - d, c])
        else:
            v = Vector([1, 0]) if math.isclose(eigenvalue, a) else Vector([0, 1])

        return v.normalize()


# ── tests (the spec — make them all pass) ────────────────────────────────────


def _vec_close(u: Vector, w: Vector, tol: float = 1e-9) -> bool:
    return all(
        math.isclose(a, b, abs_tol=tol) for a, b in zip(u.values, w.values, strict=True)
    )


if __name__ == "__main__":
    print("=== Rotation ===")
    r90 = TransformMatrix.rotation(math.pi / 2)
    assert _vec_close(r90 @ Vector([1, 0]), Vector([0.0, 1.0]))
    print("✓ rotation(π/2) sends [1,0] → [0,1]")

    print("\n=== Inherited from OperableMatrix (compounding works) ===")
    # determinant of a pure rotation is 1 (it preserves area) — method inherited,
    # not rewritten
    assert math.isclose(r90.determinant(), 1.0, abs_tol=1e-9)
    print("✓ r90.determinant() == 1.0 (inherited from lesson 02)")

    print("\n=== Composition: order matters ===")
    S = TransformMatrix.scaling(2, 1)
    R = TransformMatrix.rotation(math.pi / 2)
    v = Vector([1, 0])
    scale_then = (S @ R) @ v  # R first, then S
    rotate_then = (R @ S) @ v  # S first, then R
    print("(S@R)@v =", scale_then)
    print("(R@S)@v =", rotate_then)
    assert not _vec_close(scale_then, rotate_then), "order should matter here!"
    print("✓ (S@R)@v ≠ (R@S)@v — composition is non-commutative")

    print("\n=== Eigenvalues ===")
    diag = TransformMatrix([[2, 0], [0, 3]])
    assert sorted(diag.eigenvalues()) == [2.0, 3.0]
    sym = TransformMatrix([[2, 1], [1, 2]])
    assert sorted(sym.eigenvalues()) == [1.0, 3.0]
    print("✓ eigenvalues of diag → {2,3}, symmetric → {1,3}")

    print("\n=== Eigenvectors: A@v ≈ λ·v ===")
    for A in (diag, sym):
        for lam in A.eigenvalues():
            ev = A.eigenvector_for(lam)
            lhs = A @ ev
            rhs = Vector([lam * x for x in ev.values])
            assert _vec_close(lhs, rhs), f"A@v != λv for λ={lam}"
    print("✓ every eigenvector satisfies A@v = λ·v")

    print("\nAll tests passed! 🎉")
