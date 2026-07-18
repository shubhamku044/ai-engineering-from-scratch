# Phase 1 · Lesson 02 — Vectors, Matrices & Operations
# Extend the Matrix into the operations a neural network actually uses:
# element-wise ops, determinant, inverse, identity — then build one dense layer.
# Vector/Matrix are copied from lesson 01 into this folder — that's your code to extend.
import math

from linear_algebra import Matrix, Vector

# ── extend below ─────────────────────────────────────────────────────────────
# Vector and Matrix are now available. Add your new operations here.


class OperableMatrix(Matrix):
    """Extend the Matrix class with additional operations."""

    def __add__(self, other: "Matrix") -> "Matrix":
        """
        Perform element-wise matrix addition.

        Raises:
            ValueError: If matrix shapes differ.
        """
        if self.shape != other.shape:
            raise ValueError(
                f"Cannot add matrices of shapes {self.shape} and {other.shape}."
            )

        rows = [
            [a + b for a, b in zip(row_a, row_b, strict=True)]
            for row_a, row_b in zip(
                self.rows,
                other.rows,
                strict=True,
            )
        ]

        return Matrix(rows)

    def __mul__(self, scalar: float) -> "Matrix":
        """
        Multiply every element by a scalar.
        """
        return Matrix([[scalar * value for value in row] for row in self.rows])

    def __rmul__(self, scalar: float) -> "Matrix":
        """Support scalar * Matrix."""
        return self * scalar

    def hadamard(self, other: "Matrix") -> "Matrix":
        """
        Perform element-wise multiplication.

        Raises:
            ValueError: If shapes differ.
        """
        if self.shape != other.shape:
            raise ValueError(
                f"Cannot perform Hadamard product on shapes "
                f"{self.shape} and {other.shape}."
            )

        rows = [
            [a * b for a, b in zip(row_a, row_b, strict=True)]
            for row_a, row_b in zip(
                self.rows,
                other.rows,
                strict=True,
            )
        ]

        return Matrix(rows)

    def _minor(self, row: int, column: int) -> "OperableMatrix":
        """
        Return the minor matrix after removing one row and one column.
        """
        return OperableMatrix(
            [
                [value for c, value in enumerate(current_row) if c != column]
                for r, current_row in enumerate(self.rows)
                if r != row
            ]
        )

    def determinant(self) -> float:
        """
        Compute the determinant using recursive Laplace expansion.

        Raises:
            ValueError: If the matrix is not square.
        """
        rows, cols = self.shape

        if rows != cols:
            raise ValueError("Determinant is only defined for square matrices.")

        if rows == 1:
            return self.rows[0][0]

        if rows == 2:
            return self.rows[0][0] * self.rows[1][1] - self.rows[0][1] * self.rows[1][0]

        determinant = 0.0

        for column, value in enumerate(self.rows[0]):
            sign = (-1) ** column

            determinant += sign * value * self._minor(0, column).determinant()

        return determinant

    def inverse(self) -> "Matrix":
        """
        Compute the inverse using Gauss-Jordan elimination.

        Raises:
            ValueError:
                If the matrix is not square or is singular.
        """
        rows, cols = self.shape

        if rows != cols:
            raise ValueError("Only square matrices can be inverted.")

        if abs(self.determinant()) < 1e-9:
            raise ValueError("Matrix is singular.")

        left = [row.copy() for row in self.rows]
        right = OperableMatrix.identity(rows).rows

        for pivot in range(rows):
            pivot_value = left[pivot][pivot]

            if abs(pivot_value) < 1e-9:
                raise ValueError("Matrix is singular.")

            # Normalize pivot row
            for column in range(rows):
                left[pivot][column] /= pivot_value
                right[pivot][column] /= pivot_value

            # Eliminate other rows
            for row in range(rows):
                if row == pivot:
                    continue

                factor = left[row][pivot]

                for column in range(rows):
                    left[row][column] -= factor * left[pivot][column]
                    right[row][column] -= factor * right[pivot][column]

        return Matrix(right)

    @classmethod
    def identity(cls, n: int) -> "Matrix":
        """
        Return an n × n identity matrix.
        """
        return cls(
            [[1.0 if row == column else 0.0 for column in range(n)] for row in range(n)]
        )


def relu(vector: Vector) -> Vector:
    """
    Apply the ReLU activation function.
    """
    return Vector([max(0.0, value) for value in vector.values])


def dense_layer(
    weights: Matrix,
    inputs: Vector,
    bias: Vector,
) -> Vector:
    """
    Compute one dense neural network layer.

    output = ReLU(Wx + b)
    """
    return relu(weights @ inputs + bias)


if __name__ == "__main__":
    TOL = 1e-9

    print("=== Identity ===")

    identity = OperableMatrix.identity(3)
    print(identity)
    assert identity.rows == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    print("✓ Identity check passed.")

    print("\n=== Determinant ===")

    # 2x2:  1*4 - 2*3 = -2
    assert OperableMatrix([[1, 2], [3, 4]]).determinant() == -2

    # Diagonal 3x3:  2 * 3 * 4 = 24  (exercises the recursive _minor path)
    diagonal = OperableMatrix([[2, 0, 0], [0, 3, 0], [0, 0, 4]])
    assert math.isclose(diagonal.determinant(), 24.0, abs_tol=TOL)

    # General 3x3:  known determinant is -3
    general = OperableMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    assert math.isclose(general.determinant(), -3.0, abs_tol=TOL)

    print("✓ Determinant check passed.")

    print("\n=== Inverse ===")

    a = OperableMatrix([[4, 7], [2, 6]])
    a_inv = a.inverse()
    print("A         =", a)
    print("A inverse =", a_inv)

    # A @ A⁻¹ should be the identity matrix
    product = a @ a_inv
    for i, row in enumerate(product.rows):
        for j, value in enumerate(row):
            assert math.isclose(value, 1.0 if i == j else 0.0, abs_tol=TOL)

    print("✓ Inverse check passed.")

    print("\n=== Element-wise Operations ===")

    m1 = OperableMatrix([[1, 2], [3, 4]])
    m2 = OperableMatrix([[5, 6], [7, 8]])

    assert (m1 + m2).rows == [[6, 8], [10, 12]]  # addition
    assert (m1 * 2).rows == [[2, 4], [6, 8]]  # scalar * (right)
    assert (2 * m1).rows == [[2, 4], [6, 8]]  # scalar * (left, __rmul__)
    assert m1.hadamard(m2).rows == [[5, 12], [21, 32]]  # element-wise product

    print("✓ Element-wise operations check passed.")

    print("\n=== Dense Layer ===")

    # output = ReLU(Wx + b)
    weights = Matrix([[1, 0], [0, 1], [-1, -1]])  # 3x2
    inputs = Vector([2, 3])
    bias = Vector([0, 0, 1])

    # Wx = [2, 3, -5];  + b = [2, 3, -4];  ReLU clamps -4 → 0
    output = dense_layer(weights, inputs, bias)
    print("Output:", output)
    assert output.values == [2, 3, 0]

    print("✓ Dense layer check passed.")

    print("\nAll tests passed! 🎉")
