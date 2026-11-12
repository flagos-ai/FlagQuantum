"""Shared bounded numeric bindings and reversible output-table emission."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from fractions import Fraction
from typing import NoReturn, TypeVar

from .._fixed_word import truth_cubes
from .fixed_point import FixedPointBranches, FixedPointScalar, fixed_point_number

_Location = TypeVar("_Location", bound=Hashable)
_Fail = Callable[[str, str], NoReturn]


class FixedPointScope:
    """Keep source bindings on one correlated prepared-selector domain.

    Adapters resolve declarations and actual/formal types. These shared facts
    enforce fresh writes, initialized reads and ownership without executing a
    source function or creating a second executable representation.
    """

    def __init__(
        self,
        branches: FixedPointBranches,
        roles: Mapping[str, str],
        inputs: Mapping[str, FixedPointScalar],
    ) -> None:
        self.branches = branches
        self.roles = dict(roles)
        self.values = dict(inputs)
        self.retired: set[str] = set()
        self.zero: set[str] = set()

    def declare(self, name: str, fail: _Fail) -> None:
        if name in self.roles:
            fail("ownership", "numeric local must have a fresh name")
        self.roles[name] = "local"

    def reserve(self, name: str, fail: _Fail) -> str:
        if (
            self.roles.get(name) not in {"local", "output"}
            or name in self.values
            or name in self.retired
        ):
            fail("ownership", "quantized assignment requires a fresh local or Output")
        return name

    def read(self, name: str, fail: _Fail) -> FixedPointScalar:
        if name not in self.values:
            fail("ownership", "quantized call requires one live numeric variable")
        return self.values[name]

    def allocate(self, name: str, fail: _Fail) -> None:
        self.reserve(name, fail)
        self.values[name] = fixed_point_number(fail, Fraction())
        self.zero.add(name)

    def hadamard(self, name: str, fail: _Fail) -> None:
        if name not in self.zero:
            fail("ownership", "quantized Hadamard needs a fresh one-bit allocation")
        self.values[name] = self.branches.uniform(fail)
        self.zero.remove(name)

    def prepare_uniform(self, name: str, fail: _Fail) -> None:
        self.reserve(name, fail)
        self.values[name] = self.branches.uniform(fail)

    def drop(self, name: str, fail: _Fail) -> None:
        if self.roles.get(name) != "local":
            fail("ownership", "quantized drop requires an owned local")
        self.branches.discard(self.read(name, fail))
        self.retired.add(name)
        self.values.pop(name)
        self.zero.discard(name)


def emit_fixed_point_tables(
    reads: tuple[_Location, ...],
    outputs: Sequence[tuple[FixedPointScalar, tuple[_Location, ...]]],
    *,
    flip: Callable[[_Location], None],
    controlled: Callable[[tuple[_Location, ...], int, _Location], None],
) -> None:
    """Write fresh zero outputs with shared columns and disjoint truth cubes."""
    shared: dict[tuple[int, ...], _Location] = {}
    for value, targets in outputs:
        if len(targets) != value.width:
            raise ValueError("quantized table targets must match the inferred width")
        raw = tuple(
            int(number * (1 << value.fraction))
            for number in value.values[: 1 << len(reads)]
        )
        for bit, target in enumerate(targets):
            table = tuple((number >> bit) & 1 for number in raw)
            if not any(table):
                continue
            same = shared.get(table)
            inverse = shared.get(tuple(1 - number for number in table))
            if inverse is not None:
                flip(target)
            source = same if same is not None else inverse
            if source is not None:
                controlled((source,), 1, target)
                continue
            for cube in truth_cubes(table, len(reads)):
                if cube:
                    controls = tuple(reads[index] for index, _ in cube)
                    pattern = sum(state << bit for bit, (_, state) in enumerate(cube))
                    controlled(controls, pattern, target)
                else:
                    flip(target)
            shared[table] = target
