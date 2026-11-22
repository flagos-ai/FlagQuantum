# SPDX-License-Identifier: Apache-2.0
# Adapted from Kickmix operation_statistics.pybind.cc/.h at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Read-only projections of exact physical instruction counter snapshots."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TypeVar

from .kmb_operations import OPERATION_CODES
from .kmx_circuit import KmxCircuit

_MASK = (1 << 64) - 1
_GROUP_NAMES = (
    "NEG",
    "BIT_INVERT",
    "BIT_STORE0",
    "BIT_STORE1",
    "X",
    "Z",
    "R",
    "HMR",
    "CX",
    "CZ",
    "SWAP",
    "CSWAP",
    "CCX",
    "CCZ",
    "Z_POW",
    "INFO_TAG",
    "DEBUG_PRINT",
    "POP_CONDITION",
    "PUSH_CONDITION",
)
_GROUPS = {
    name.encode(): tuple(
        code
        for opcode, code in OPERATION_CODES.items()
        if opcode.removesuffix("_IF") == name
        or name == "DEBUG_PRINT"
        and opcode.startswith("DEBUG_PRINT_")
    )
    for name in _GROUP_NAMES
}
_Default = TypeVar("_Default")


def _key_bytes(key: str | bytes | bytearray) -> bytes:
    if isinstance(key, str):
        return key.encode()
    if isinstance(key, (bytes, bytearray)):
        return bytes(key)
    raise TypeError("Operation count key must be str, bytes or bytearray")


@dataclass(frozen=True)
class PhysicalOperationStatistics:
    """Retain raw opcode identity and exact phase levels without execution."""

    raw_op_counts: tuple[int, ...] = (0,) * 256
    z_pow_level_counts: tuple[int, ...] = (0,) * 129

    def __post_init__(self) -> None:
        if not isinstance(self.raw_op_counts, tuple) or not isinstance(
            self.z_pow_level_counts, tuple
        ):
            raise TypeError("Physical snapshot storage must be immutable tuples")
        if len(self.raw_op_counts) != 256 or len(self.z_pow_level_counts) != 129:
            raise ValueError(
                "Physical snapshots require 256 opcode and 129 level counts"
            )
        if any(
            type(value) is not int or not 0 <= value <= _MASK
            for value in (*self.raw_op_counts, *self.z_pow_level_counts)
        ):
            raise ValueError("Physical counts must be unsigned 64-bit integers")

    @classmethod
    def from_circuit(cls, circuit: KmxCircuit) -> "PhysicalOperationStatistics":
        """Count stored instructions independently of their activation."""
        return cls.from_counts(
            (
                (
                    record.opcode
                    + (
                        "_IF"
                        if record.condition_bit is not None
                        and record.opcode != "PUSH_CONDITION"
                        else ""
                    ),
                    1,
                )
                for record in circuit
            ),
            (
                (record.angle.clifford_hierarchy_level(), 1)
                for record in circuit
                if record.angle is not None
            ),
        )

    @classmethod
    def from_counts(
        cls,
        operations: Iterable[tuple[str, int]],
        levels: Iterable[tuple[int, int]],
    ) -> "PhysicalOperationStatistics":
        """Project owned primitive counter pairs into an immutable snapshot."""
        raw = [0] * 256
        rotations = [0] * 129
        for name, count in operations:
            if type(count) is not int or not 0 <= count <= _MASK:
                raise ValueError("Physical counts must be unsigned 64-bit integers")
            code = OPERATION_CODES[name]
            raw[code] = (raw[code] + count) & _MASK
        for level, count in levels:
            if type(level) is not int or not 0 <= level <= 128:
                raise ValueError("Rotation hierarchy level must be between 0 and 128")
            if type(count) is not int or not 0 <= count <= _MASK:
                raise ValueError("Physical counts must be unsigned 64-bit integers")
            rotations[level] = (rotations[level] + count) & _MASK
        return cls(tuple(raw), tuple(rotations))

    @property
    def num_toffoli_like(self) -> int:
        return sum(self[name] for name in ("CCX", "CCZ", "CSWAP")) & _MASK

    @property
    def num_non_clifford(self) -> int:
        return (self.num_toffoli_like + self.num_z_pow(3)) & _MASK

    @property
    def num_t_like(self) -> int:
        return self.z_pow_level_counts[3]

    @property
    def num_instructions(self) -> int:
        return sum(self.raw_op_counts) & _MASK

    def num_z_pow(
        self,
        min_clifford_hierarchy_level: int = 0,
        max_clifford_hierarchy_level: int | None = None,
    ) -> int:
        """Count rotations in the inclusive source-defined clipped range."""
        for level in (min_clifford_hierarchy_level, max_clifford_hierarchy_level):
            if level is not None and (
                not isinstance(level, int) or not -(1 << 63) <= level < 1 << 63
            ):
                raise TypeError("Rotation range bounds must be signed 64-bit integers")
        low = max(0, min_clifford_hierarchy_level)
        high = min(
            128,
            (
                128
                if max_clifford_hierarchy_level is None
                else max_clifford_hierarchy_level
            ),
        )
        return sum(self.z_pow_level_counts[low : max(low, high + 1)]) & _MASK

    @property
    def z_pow_clifford_hierarchy_counts(self) -> dict[int, int]:
        return {
            level: count for level, count in enumerate(self.z_pow_level_counts) if count
        }

    def __getitem__(self, key: str | bytes | bytearray) -> int:
        encoded = _key_bytes(key)
        codes = _GROUPS.get(encoded)
        if codes is None:
            raise KeyError(encoded.decode())
        return sum(self.raw_op_counts[code] for code in codes) & _MASK

    def get(
        self, key: str | bytes | bytearray, default: _Default | None = None
    ) -> int | _Default | None:
        codes = _GROUPS.get(_key_bytes(key), ())
        count = sum(self.raw_op_counts[code] for code in codes) & _MASK
        return count if count else default

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and bool(self.get(key, 0))

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def keys(self) -> list[str]:
        return [name for name in _GROUP_NAMES if self[name]]

    def values(self) -> list[int]:
        return [self[name] for name in self.keys()]

    def items(self) -> list[tuple[str, int]]:
        return [(name, self[name]) for name in self.keys()]

    def __repr__(self) -> str:
        return (
            f"<PhysicalOperationStatistics num_instructions={self.num_instructions}"
            f" num_toffoli_like={self.num_toffoli_like} num_t_like={self.num_t_like}>"
        )
