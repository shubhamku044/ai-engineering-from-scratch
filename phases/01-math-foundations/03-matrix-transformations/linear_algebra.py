# Phase 1 · Lesson 01 — Linear Algebra Intuition
# From scratch, NO numpy. Vectors and matrices are just lists of numbers +
# operations. Build the operations; feel what they mean geometrically.
import math
from typing import Union, overload


class Vector:
    def __init__(self, values: list[float]) -> None:
        """create a vector

        Args:
            values (list[float]): compoents of the vector

        Raises:
            ValueError: if the vector has no values
        """
        if not values:
            raise ValueError("Vector must have at least one value.")
        self.values = values

    def __len__(self) -> int:
        """to get dimension of the vector

        Returns:
            int: dimension of the vector
        """

        return len(self.values)

    def __add__(self, other: "Vector") -> "Vector":
        self._validate_dimension(other)
        return Vector(
            [v1 + v2 for v1, v2 in zip(self.values, other.values, strict=True)]
        )

    def __sub__(self, other: "Vector") -> "Vector":
        self._validate_dimension(other)
        return Vector(
            [v1 - v2 for v1, v2 in zip(self.values, other.values, strict=True)]
        )

    def dot(self, other: "Vector") -> float:
        self._validate_dimension(other)
        return sum(v1 * v2 for v1, v2 in zip(self.values, other.values, strict=True))

    def magnitude(self) -> float:
        return math.sqrt(sum(v**2 for v in self.values))

    def normalize(self) -> "Vector":
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize a zero vector.")
        return Vector([v / mag for v in self.values])

    def cosine_similarity(self, other: "Vector") -> float:
        self._validate_dimension(other)
        mag_self = self.magnitude()
        mag_other = other.magnitude()
        if mag_self == 0 or mag_other == 0:
            raise ValueError("Cannot compute cosine similarity with a zero vector.")
        return self.dot(other) / (mag_self * mag_other)

    def angle_between(self, other: "Vector") -> float:
        cosine = self.cosine_similarity(other)

        # Clamp because of floating point arthemetic
        # may produce values like 1.0000000000000002 or -1.0000000000000002
        cosine = max(-1.0, min(1.0, cosine))

        return math.degrees(math.acos(cosine))

    def __repr__(self) -> str:
        return f"Vector({self.values})"

    def _validate_dimension(self, other: "Vector") -> None:
        """
        Ensure both vectors have equal dimensions.
        """
        if len(self) != len(other):
            raise ValueError(f"Dimension mismatch: {len(self)} != {len(other)}")

    @staticmethod
    def project(a: "Vector", b: "Vector") -> "Vector":
        """
        Project vector a onto vector b.

        Args:
            a: The vector to be projected.
            b: The vector onto which a is projected.

        Returns:
            The projection of a onto b.

        Raises:
            ValueError: If b is a zero vector or if dimensions do not match.
        """
        denominator = b.dot(b)
        if math.isclose(denominator, 0.0):
            raise ValueError("Cannot project onto a zero vector.")

        scalar_projection = a.dot(b) / denominator
        return Vector([scalar_projection * v for v in b.values])

    @staticmethod
    def gram_schmidt(vectors: list["Vector"]) -> list["Vector"]:
        """
        Convert independent vectors into an orthonormal basis.
        """
        basis: list[Vector] = []

        for vector in vectors:
            orthogonal = Vector(vector.values.copy())

            for basis_vector in basis:
                orthogonal = orthogonal - Vector.project(orthogonal, basis_vector)

            if math.isclose(orthogonal.magnitude(), 0.0, abs_tol=1e-9):
                raise ValueError("Vectors are linearly dependent.")

            basis.append(orthogonal.normalize())

        return basis

    @staticmethod
    def is_linearly_independent(vectors: list["Vector"]) -> bool:
        """
        Return whether the vectors are linearly independent.
        """
        try:
            Vector.gram_schmidt(vectors)
            return True
        except ValueError:
            return False


class Matrix:
    """Represents a mathematical matrix."""

    def __init__(self, rows: list[list[float]]) -> None:
        """
        Create a matrix.

        Args:
            rows: Matrix rows.

        Raises:
            ValueError: If the matrix is empty or rows have different lengths.
        """
        if not rows:
            raise ValueError("Matrix cannot be empty.")

        row_length = len(rows[0])

        if row_length == 0:
            raise ValueError("Matrix rows cannot be empty.")

        if any(len(row) != row_length for row in rows):
            raise ValueError("All rows must have the same length.")

        self.rows = rows

    @property
    def shape(self) -> tuple[int, int]:
        """Return the matrix shape as (rows, columns)."""
        return len(self.rows), len(self.rows[0])

    def transpose(self) -> "Matrix":
        """
        Return the transpose of the matrix.
        """
        return Matrix([list(column) for column in zip(*self.rows, strict=True)])

    @overload
    def __matmul__(self, other: Vector) -> Vector: ...

    @overload
    def __matmul__(self, other: "Matrix") -> "Matrix": ...

    def __matmul__(self, other: Union["Matrix", Vector]) -> Union["Matrix", Vector]:
        """
        Perform matrix multiplication.

        Supports:
            Matrix @ Vector
            Matrix @ Matrix

        Raises:
            ValueError: If dimensions are incompatible.
            TypeError: If the operand type is unsupported.
        """
        _, cols = self.shape

        # Matrix @ Vector
        if isinstance(other, Vector):
            if cols != len(other):
                raise ValueError(
                    f"Cannot multiply matrix of shape {self.shape} "
                    f"with vector of dimension {len(other)}."
                )

            result = [
                sum(
                    value * vector
                    for value, vector in zip(
                        row,
                        other.values,
                        strict=True,
                    )
                )
                for row in self.rows
            ]

            return Vector(result)

        # Matrix @ Matrix
        if isinstance(other, Matrix):  # type: ignore
            other_rows, _ = other.shape

            if cols != other_rows:
                raise ValueError(
                    f"Cannot multiply matrices with shapes "
                    f"{self.shape} and {other.shape}."
                )

            other_transposed = other.transpose()

            product_rows: list[list[float]] = []

            for row in self.rows:
                new_row: list[float] = []

                for column in other_transposed.rows:
                    value = sum(
                        a * b
                        for a, b in zip(
                            row,
                            column,
                            strict=True,
                        )
                    )

                    new_row.append(value)

                product_rows.append(new_row)

            return Matrix(product_rows)

        raise TypeError(f"Unsupported operand type: {type(other).__name__}")

    def __repr__(self) -> str:
        """Return a readable representation of the matrix."""
        return f"Matrix({self.rows})"


if __name__ == "__main__":
    print("=== Vector Operations ===")

    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])

    print("v1:", v1)
    print("v2:", v2)
    print("v1 + v2:", v1 + v2)
    print("v1 - v2:", v1 - v2)
    print("v1 · v2:", v1.dot(v2))
    print("Magnitude of v1:", v1.magnitude())
    print("Normalized v1:", v1.normalize())
    print("Cosine similarity:", v1.cosine_similarity(v2))
    print("Angle between v1 and v2:", v1.angle_between(v2), "degrees")

    print("\n=== Sanity Checks ===")

    assert Vector([1, 2, 3]).dot(Vector([4, 5, 6])) == 32
    assert Vector([3, 4]).magnitude() == 5.0
    assert round(Vector([1, 0]).angle_between(Vector([0, 1])), 5) == 90.0

    print("✓ Dot product check passed.")
    print("✓ Magnitude check passed.")
    print("✓ Angle check passed.")

    print("\n=== Matrix @ Vector ===")

    matrix = Matrix(
        [
            [1, 2],
            [3, 4],
        ]
    )

    vector = Vector([5, 6])

    print(matrix)
    print(vector)
    print(matrix @ vector)

    print("\n=== Matrix @ Matrix ===")

    matrix_a = Matrix(
        [
            [1, 2],
            [3, 4],
        ]
    )

    matrix_b = Matrix(
        [
            [5, 6],
            [7, 8],
        ]
    )

    product = matrix_a @ matrix_b

    print("A =", matrix_a)
    print("B =", matrix_b)
    print("A @ B =", product)

    # Hand-check:
    # First element = (1 * 5) + (2 * 7) = 19
    assert product.rows[0][0] == 19

    print("✓ Matrix multiplication check passed.")

    print("\n=== Matrix Transpose ===")

    rectangular = Matrix(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    transposed = rectangular.transpose()

    print("Original:", rectangular)
    print("Transpose:", transposed)

    assert rectangular.shape == (2, 3)
    assert transposed.shape == (3, 2)

    print("✓ Transpose check passed.")

    print("\n=== Rotation Matrix ===")

    # Prediction:
    # A 90° counter-clockwise rotation should send
    # Vector([3, 1]) -> Vector([-1, 3])

    rotation = Matrix(
        [
            [0, -1],
            [1, 0],
        ]
    )

    point = Vector([3, 1])
    rotated = rotation @ point

    print("Original:", point)
    print("Rotated :", rotated)

    assert rotated.values == [-1, 3]

    print("✓ Rotation check passed.")

    print("\n=== Projection ===")

    a = Vector([3, 4])
    b = Vector([1, 0])

    projection = Vector.project(a, b)

    print("Projection:", projection)

    assert projection.values == [3.0, 0.0]

    print("✓ Projection check passed.")

    print("\n=== Gram-Schmidt ===")

    basis = Vector.gram_schmidt(
        [
            Vector([1, 0, 0]),
            Vector([1, 1, 0]),
            Vector([1, 1, 1]),
        ]
    )

    for vector in basis:
        print(vector)

    TOL = 1e-9

    for vector in basis:
        assert math.isclose(
            vector.magnitude(),
            1.0,
            abs_tol=TOL,
        )

    for i in range(len(basis)):
        for j in range(i + 1, len(basis)):
            assert math.isclose(
                basis[i].dot(basis[j]),
                0.0,
                abs_tol=TOL,
            )

    print("✓ Gram-Schmidt check passed.")

    print("\n=== Linear Independence ===")

    independent = [
        Vector([1, 0, 0]),
        Vector([0, 1, 0]),
        Vector([0, 0, 1]),
    ]

    dependent = [
        Vector([1, 0, 0]),
        Vector([0, 1, 0]),
        Vector([2, 1, 0]),
    ]

    assert Vector.is_linearly_independent(independent)
    assert not Vector.is_linearly_independent(dependent)

    print("✓ Linear independence check passed.")

    print("\nAll tests passed! 🎉")
