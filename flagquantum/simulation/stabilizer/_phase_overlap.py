"""Private finite CH feasibility prototype; no production route imports this.

Mathematical authority: Bravyi et al., arXiv:1808.00128v2, Section 4.1,
Eqs. 42-56. The approved implementation notes describe orientation and cost.
Only masks and a validated word are retained, never a dense state or unitary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_Word = tuple[tuple[str, tuple[int, ...]], ...]
_ROOTS = (1 + 0j, 1j, -1 + 0j, -1j)
_ATOL = 1e-12


def _unit_phase(value: complex) -> None:
    if (
        type(value) is not complex
        or not math.isfinite(value.real)
        or not math.isfinite(value.imag)
        or abs(abs(value) - 1) > _ATOL
    ):
        raise ValueError("private CH phase must be finite complex unit modulus")


def _validate_word(n: int, word: _Word) -> None:
    if type(n) is not int or n not in (1, 2, 3):
        raise ValueError("private CH width is restricted to one through three wires")
    if type(word) is not tuple or len(word) > 22:
        raise ValueError("private CH requires a tuple word of at most 22 instructions")
    for instruction in word:
        if type(instruction) is not tuple or len(instruction) != 2:
            raise ValueError("private CH requires gate and wire tuples")
        gate, wires = instruction
        if type(gate) is not str or gate not in ("h", "s", "x", "cx"):
            raise ValueError("private CH accepts only literal H/S/X/CX inputs")
        if (
            type(wires) is not tuple
            or len(wires) != (2 if gate == "cx" else 1)
            or any(type(wire) is not int or not 0 <= wire < n for wire in wires)
            or len(set(wires)) != len(wires)
        ):
            raise ValueError("private CH requires distinct valid integer wires")


@dataclass(frozen=True)
class _CliffordState:
    """Owned phase-sensitive CH masks; construct through from_word only."""

    n: int
    f: tuple[int, ...]
    g: tuple[int, ...]
    m: tuple[int, ...]
    gamma: tuple[int, ...]
    v: int
    s: int
    omega: complex
    word: _Word
    scalar_phase: complex

    @classmethod
    def from_word(
        cls, n: int, word: _Word, *, scalar_phase: complex = 1 + 0j
    ) -> _CliffordState:
        _validate_word(n, word)
        _unit_phase(scalar_phase)
        builder = _CHBuilder(n)
        for gate, wires in word:
            builder.apply(gate, wires)
        return cls(
            n,
            tuple(builder.f),
            tuple(builder.g),
            tuple(builder.m),
            tuple(builder.gamma),
            builder.v,
            builder.s,
            builder.omega * scalar_phase,
            word,
            scalar_phase,
        )


class _CHBuilder:
    """Mutable polynomial scratch for one word, never a retained dense state."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.f = [1 << (n - 1 - wire) for wire in range(n)]
        self.g = self.f[:]
        self.m = [0] * n
        self.gamma = [0] * n
        self.v = 0
        self.s = 0
        self.omega = 1 + 0j

    @classmethod
    def from_state(cls, state: _CliffordState) -> _CHBuilder:
        builder = cls(state.n)
        builder.f = list(state.f)
        builder.g = list(state.g)
        builder.m = list(state.m)
        builder.gamma = list(state.gamma)
        builder.v = state.v
        builder.s = state.s
        builder.omega = state.omega
        return builder

    def apply(self, gate: str, wires: tuple[int, ...]) -> None:
        q = wires[0]
        if gate == "h":
            self._h(q)
        elif gate == "s":
            self.m[q] ^= self.g[q]
            self.gamma[q] = (self.gamma[q] - 1) % 4
        elif gate == "cx":
            r = wires[1]
            sign = (self.m[q] & self.f[r]).bit_count() % 2
            self.gamma[q] = (self.gamma[q] + self.gamma[r] + 2 * sign) % 4
            self.g[r] ^= self.g[q]
            self.f[q] ^= self.f[r]
            self.m[q] ^= self.m[r]
        elif gate == "x":
            self._h(q)
            self.apply("s", (q,))
            self.apply("s", (q,))
            self._h(q)
        else:
            raise ValueError("invalid private CH update")

    def _right(self, gate: str, q: int, r: int = 0) -> None:
        qmask = 1 << (self.n - 1 - q)
        rmask = 1 << (self.n - 1 - r)
        for row in range(self.n):
            fq = bool(self.f[row] & qmask)
            if gate == "s":
                if fq:
                    self.m[row] ^= qmask
                    self.gamma[row] = (self.gamma[row] - 1) % 4
            elif gate == "cx":
                if self.g[row] & rmask:
                    self.g[row] ^= qmask
                if fq:
                    self.f[row] ^= rmask
                if self.m[row] & rmask:
                    self.m[row] ^= qmask
            elif gate == "cz":
                fr = bool(self.f[row] & rmask)
                if fr:
                    self.m[row] ^= qmask
                if fq:
                    self.m[row] ^= rmask
                self.gamma[row] = (self.gamma[row] + 2 * fq * fr) % 4
            else:
                raise ValueError("invalid private CH right multiplication")

    def _h(self, q: int) -> None:
        t = self.s ^ (self.g[q] & self.v)
        u = self.s ^ (self.f[q] & ~self.v) ^ (self.m[q] & self.v)
        alpha = (self.g[q] & ~self.v & self.s).bit_count() % 2
        beta = (
            (self.m[q] & ~self.v & self.s).bit_count()
            + (self.f[q] & self.v & (self.m[q] ^ self.s)).bit_count()
        ) % 2
        if t == u:
            self.omega *= ((-1) ** alpha + _ROOTS[self.gamma[q]] * (-1) ** beta) / (
                math.sqrt(2)
            )
            self.s = t
            return
        delta = (self.gamma[q] + 2 * (alpha + beta)) % 4
        difference = t ^ u
        no_h = difference & ~self.v
        pivot_mask = 1 << ((no_h or difference).bit_length() - 1)
        pivot = self.n - pivot_mask.bit_length()
        hadamard = bool(self.v & pivot_mask)
        for wire in range(self.n):
            mask = 1 << (self.n - 1 - wire)
            if wire == pivot or not difference & mask:
                continue
            if no_h:
                self._right("cz" if self.v & mask else "cx", pivot, wire)
            else:
                self._right("cx", wire, pivot)
        factor, phase_gate, h_gate, bit = _merge_one_qubit(
            hadamard, bool(t & pivot_mask), delta
        )
        if phase_gate:
            self._right("s", pivot)
        self.s = (u ^ pivot_mask) if t & pivot_mask else t
        self.s = (self.s & ~pivot_mask) | (pivot_mask if bit else 0)
        self.v = (self.v & ~pivot_mask) | (pivot_mask if h_gate else 0)
        self.omega *= (-1) ** alpha * factor


def _merge_one_qubit(h: bool, t: bool, delta: int) -> tuple[complex, int, int, int]:
    # Constant-size algebra for H^h (|t> + i^delta |1-t>) / sqrt(2).
    first = (_ROOTS[delta] if t else 1 + 0j) / math.sqrt(2)
    second = (1 + 0j if t else _ROOTS[delta]) / math.sqrt(2)
    if h:
        first, second = (first + second) / math.sqrt(2), (first - second) / math.sqrt(2)
    if abs(first) < _ATOL:
        return second, 0, 0, 1
    if abs(second) < _ATOL:
        return first, 0, 0, 0
    relative = second / first
    power = next(
        (index for index, root in enumerate(_ROOTS) if abs(relative - root) < _ATOL),
        None,
    )
    if power is None:
        raise ValueError("private single-qubit phase is not a Clifford root")
    return first * math.sqrt(2), power % 2, 1, power // 2


def _validate_state(state: _CliffordState) -> None:
    _validate_word(state.n, state.word)
    _unit_phase(state.scalar_phase)
    _unit_phase(state.omega)
    bound = 1 << state.n
    for values, ceiling in (
        (state.f, bound),
        (state.g, bound),
        (state.m, bound),
        (state.gamma, 4),
    ):
        if type(values) is not tuple or len(values) != state.n:
            raise ValueError("private CH row shape is invalid")
        if any(type(value) is not int or not 0 <= value < ceiling for value in values):
            raise ValueError("private CH rows require bounded integers")
    if any(
        type(value) is not int or not 0 <= value < bound for value in (state.v, state.s)
    ):
        raise ValueError("private CH bit masks are invalid")
    for row in range(state.n):
        for column in range(state.n):
            if (state.f[row] & state.g[column]).bit_count() % 2 != int(row == column):
                raise ValueError("private CH conjugation rows are not dual")


def _overlap(bra: _CliffordState, ket: _CliffordState) -> complex:
    """Compute bra-left complex overlap by polynomial Clifford-word replay."""
    _validate_state(bra)
    _validate_state(ket)
    if bra.n != ket.n:
        raise ValueError("private CH overlap requires equal widths")
    scratch = _CHBuilder.from_state(ket)
    for gate, wires in reversed(bra.word):
        for _ in range(3 if gate == "s" else 1):
            scratch.apply(gate, wires)
    if scratch.s & ~scratch.v:
        return 0j
    return (
        bra.scalar_phase.conjugate()
        * scratch.omega
        * math.pow(2, -scratch.v.bit_count() / 2)
    )
