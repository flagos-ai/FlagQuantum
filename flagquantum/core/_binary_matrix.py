# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix util/gf2_matrix.cc and gf2_matrix.h.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.
# Immutable integer rows replace mutable packed 64-bit word storage.

"""Private binary matrix value algebra shared by field synthesis and metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryMatrix:
    """Retain shape and coefficient order, including empty rectangular values.

    Bit c of packed_rows[r] is entry (r, c). Integer rows preserve the source
    packed representation without a word-size limit. Methods return new values.
    Only polynomial-vector application has the source 1024-coordinate bound.
    """

    packed_rows: tuple[int, ...] = ()
    columns: int = 0

    def __post_init__(self) -> None:
        if type(self.columns) is not int or self.columns < 0:
            raise ValueError("Binary matrix column count must be nonnegative")
        if type(self.packed_rows) is not tuple or any(
            type(row) is not int or row < 0 or row.bit_length() > self.columns
            for row in self.packed_rows
        ):
            raise ValueError("Binary matrix rows must fit the declared column count")

    @property
    def rows(self) -> int:
        return len(self.packed_rows)

    @property
    def row_words(self) -> int:
        return (self.columns + 63) // 64

    @classmethod
    def identity(cls, size: int) -> BinaryMatrix:
        return cls(tuple(1 << index for index in range(size)), size)

    @classmethod
    def from_dense(cls, values: list[list[int]]) -> BinaryMatrix:
        columns = len(values[0]) if values else 0
        if any(
            len(row) != columns or any(bit not in (0, 1) for bit in row)
            for row in values
        ):
            raise ValueError(
                "Binary matrix dense rows must be rectangular binary values"
            )
        return cls(
            tuple(
                sum(bit << column for column, bit in enumerate(row)) for row in values
            ),
            columns,
        )

    def dense(self) -> list[list[int]]:
        return [
            [(row >> column) & 1 for column in range(self.columns)]
            for row in self.packed_rows
        ]

    def get(self, row: int, column: int) -> bool:
        if not 0 <= row < self.rows or not 0 <= column < self.columns:
            raise ValueError("Binary matrix coordinate is out of range")
        return bool((self.packed_rows[row] >> column) & 1)

    def set(self, row: int, column: int, value: bool) -> BinaryMatrix:
        self.get(row, column)
        result = list(self.packed_rows)
        mask = 1 << column
        result[row] = (result[row] & ~mask) | (mask if value else 0)
        return BinaryMatrix(tuple(result), self.columns)

    def xor_bit(self, row: int, column: int) -> BinaryMatrix:
        return self.set(row, column, not self.get(row, column))

    def xor_row_into(self, source: int, target: int) -> BinaryMatrix:
        if not 0 <= source < self.rows or not 0 <= target < self.rows:
            raise ValueError("Binary matrix row is out of range")
        result = list(self.packed_rows)
        result[target] ^= result[source]
        return BinaryMatrix(tuple(result), self.columns)

    def swap_rows(self, first: int, second: int) -> BinaryMatrix:
        if not 0 <= first < self.rows or not 0 <= second < self.rows:
            raise ValueError("Binary matrix row is out of range")
        result = list(self.packed_rows)
        result[first], result[second] = result[second], result[first]
        return BinaryMatrix(tuple(result), self.columns)

    def multiply(self, other: BinaryMatrix) -> BinaryMatrix:
        if self.columns != other.rows:
            raise ValueError("Binary matrix product dimensions do not match")
        result = []
        for row in self.packed_rows:
            value = 0
            while row:
                column = (row & -row).bit_length() - 1
                row &= row - 1
                value ^= other.packed_rows[column]
            result.append(value)
        return BinaryMatrix(tuple(result), other.columns)

    def apply(self, value: int) -> int:
        if self.rows > 1024 or self.columns > 1024:
            raise ValueError("Binary polynomial matrix dimension exceeds 1024")
        if type(value) is not int or not 0 <= value < 1 << 1024:
            raise ValueError("Binary polynomial vector must fit 1024 coefficient bits")
        return sum(
            ((row & value).bit_count() & 1) << index
            for index, row in enumerate(self.packed_rows)
        )

    def transposed(self) -> BinaryMatrix:
        result = [0] * self.columns
        for index, row in enumerate(self.packed_rows):
            while row:
                column = (row & -row).bit_length() - 1
                row &= row - 1
                result[column] |= 1 << index
        return BinaryMatrix(tuple(result), self.rows)

    def plu(self) -> tuple[BinaryMatrix, BinaryMatrix, BinaryMatrix] | None:
        """Return P,L,U with self=P*L*U, retaining the source first-pivot order."""
        return self._plu(require_invertible=True)

    def plu_with_zero_pivots(
        self,
    ) -> tuple[BinaryMatrix, BinaryMatrix, BinaryMatrix] | None:
        """Retain PLU factors even when a square matrix has missing pivots.

        This matches the factorization consumed by literal source synthesis;
        it does not assert that its subsequent reversible gates implement a
        singular linear map. The ordinary plu method keeps its failure sentinel.
        """
        return self._plu(require_invertible=False)

    def _plu(
        self, *, require_invertible: bool
    ) -> tuple[BinaryMatrix, BinaryMatrix, BinaryMatrix] | None:
        if self.rows != self.columns:
            return None
        values = list(self.packed_rows)
        permutation = list(range(self.rows))
        for column in range(self.columns):
            pivot = next(
                (row for row in range(column, self.rows) if values[row] >> column & 1),
                None,
            )
            if pivot is None:
                if require_invertible:
                    return None
                continue
            values[column], values[pivot] = values[pivot], values[column]
            permutation[column], permutation[pivot] = (
                permutation[pivot],
                permutation[column],
            )
            # Keep each multiplier below the diagonal for the final L factor.
            high = values[column] & ~((1 << (column + 1)) - 1)
            for row in range(column + 1, self.rows):
                if values[row] >> column & 1:
                    values[row] ^= high
        lower = tuple(
            (value & ((1 << row) - 1)) | (1 << row) for row, value in enumerate(values)
        )
        upper = tuple(value & ~((1 << row) - 1) for row, value in enumerate(values))
        permuted = [0] * self.rows
        for column, row in enumerate(permutation):
            permuted[row] = 1 << column
        return (
            BinaryMatrix(tuple(permuted), self.columns),
            BinaryMatrix(lower, self.columns),
            BinaryMatrix(upper, self.columns),
        )

    def inverse(self) -> BinaryMatrix:
        """Return the source Gauss-Jordan inverse or its empty failure sentinel."""
        if self.rows != self.columns:
            return BinaryMatrix()
        width = self.columns
        values = [
            row | (1 << (width + index)) for index, row in enumerate(self.packed_rows)
        ]
        for column in range(width):
            pivot = next(
                (row for row in range(column, self.rows) if values[row] >> column & 1),
                None,
            )
            if pivot is None:
                return BinaryMatrix()
            values[column], values[pivot] = values[pivot], values[column]
            for row in range(self.rows):
                if row != column and values[row] >> column & 1:
                    values[row] ^= values[column]
        return BinaryMatrix(tuple(row >> width for row in values), width)

    def rank(self) -> int:
        values = list(self.packed_rows)
        result = 0
        for column in range(self.columns):
            if result == self.rows:
                break
            pivot = next(
                (row for row in range(result, self.rows) if values[row] >> column & 1),
                None,
            )
            if pivot is None:
                continue
            values[result], values[pivot] = values[pivot], values[result]
            for row in range(self.rows):
                if row != result and values[row] >> column & 1:
                    values[row] ^= values[result]
            result += 1
        return result

    def __str__(self) -> str:
        return "\n".join(
            "".join("1" if row >> column & 1 else "." for column in range(self.columns))
            for row in self.packed_rows
        )
