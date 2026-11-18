"""Symbolic support proof for independent measured-And erasure histories.

This proves a channel identity, not a statevector simulation. Boolean support
is represented by a reduced ordered Boolean decision graph. Rotations may
enlarge the support conservatively; a failed obligation is never ignored.
"""

from math import isfinite
from numbers import Real

from flagquantum.compiler.registers.primitives.and_compute import (
    and_compute_ir,
    and_uncompute_ir,
)
from flagquantum.compiler.registers.primitives.controlled_swap import (
    controlled_swap_bit_ir,
)
from flagquantum.core.ir import Instruction

ZERO, ONE = 0, 1


class _BooleanSupport:
    """Canonical Boolean functions without expanding comparator monomials."""

    def __init__(self) -> None:
        self.nodes: list[tuple[int, int, int]] = [(0, 0, 0), (0, 1, 1)]
        self.unique: dict[tuple[int, int, int], int] = {}
        self.cache: dict[tuple[bool, int, int], int] = {}

    def node(self, variable: int, low: int, high: int) -> int:
        if low == high:
            return low
        key = (variable, low, high)
        if key not in self.unique:
            self.unique[key] = len(self.nodes)
            self.nodes.append(key)
        return self.unique[key]

    def variable(self, variable: int) -> int:
        return self.node(variable, ZERO, ONE)

    def apply(self, xor: bool, left: int, right: int) -> int:
        left, right = sorted((left, right))
        if left == right:
            return ZERO if xor else left
        if left == ZERO:
            return right if xor else ZERO
        if left == ONE and not xor:
            return right
        key = (xor, left, right)
        if key in self.cache:
            return self.cache[key]
        variable = min(self.nodes[n][0] for n in (left, right) if n > ONE)
        lo_left, hi_left = (
            self.nodes[left][1:]
            if left > ONE and self.nodes[left][0] == variable
            else (left, left)
        )
        lo_right, hi_right = (
            self.nodes[right][1:]
            if right > ONE and self.nodes[right][0] == variable
            else (right, right)
        )
        result = self.node(
            variable,
            self.apply(xor, lo_left, lo_right),
            self.apply(xor, hi_left, hi_right),
        )
        self.cache[key] = result
        return result

    def multiply(self, left: int, right: int) -> int:
        return self.apply(False, left, right)

    def xor(self, left: int, right: int) -> int:
        return self.apply(True, left, right)


def _matrix_changed_wires(gate: Instruction) -> tuple[int, ...]:
    import torch

    if gate.matrix is None:
        raise AssertionError("Matrix support requires a declared operand")
    matrix = torch.as_tensor(gate.matrix, dtype=torch.complex128)
    size = 1 << len(gate.wires)
    if matrix.shape != (size, size) or not torch.isfinite(matrix).all():
        raise AssertionError("Certificate matrix must be finite and match its operands")
    if not torch.allclose(
        matrix.mH @ matrix, torch.eye(size, dtype=matrix.dtype), atol=1e-10, rtol=0
    ):
        raise AssertionError("Certificate matrix must be unitary")
    rows, columns = torch.nonzero(matrix != 0, as_tuple=True)
    # Exact structural zeros only: even a tiny nonzero transition changes support.
    return tuple(
        wire
        for offset, wire in enumerate(gate.wires)
        if bool(torch.any(((rows ^ columns) >> (len(gate.wires) - 1 - offset)) & 1))
    )


def certify_erasure_histories(
    gates: tuple[Instruction, ...], *, n_wires: int, zero_wires: tuple[int, ...]
) -> int:
    """Prove every branch is the same operator times 2**(-events/2).

    Initial zero_wires must be clean, and are proved clean at return. All other
    input wires are arbitrary, including entanglement with external spectators.
    Exact four-T compute requires a zero target. Its measured inverse requires
    target == control AND control immediately before erasure. The accepted local
    primitive has positive scalar 1/sqrt(2) in either outcome. Other gates are
    unconditional unitaries. This identity extends by linearity to all complex
    amplitudes in the admitted input subspace; no measurement history is sampled.
    """
    if type(n_wires) is not int or n_wires < 1:
        raise ValueError("Certificate width must be a positive integer")
    if any(gate.matrix is not None and gate.name != "any" for gate in gates):
        raise AssertionError("A matrix cannot override the certificate's named gates")
    if len(set(zero_wires)) != len(zero_wires) or any(
        w < 0 or w >= n_wires for w in zero_wires
    ):
        raise ValueError("Certificate clean wires must be distinct and in range")
    support = _BooleanSupport()
    if any(
        len(set(gate.wires)) != len(gate.wires)
        or any(type(w) is not int or not 0 <= w < n_wires for w in gate.wires)
        for gate in gates
    ):
        raise ValueError("Certificate operands must be distinct and in range")
    values = [ZERO if w in zero_wires else support.variable(w) for w in range(n_wires)]
    fresh, events, i = n_wires, 0, 0
    while i < len(gates):
        gate = gates[i]
        if gate.name == "cx" and i + 2 < len(gates) and gates[i + 1].name == "cx":
            right, left = gate.wires
            control, target = gates[i + 1].wires
            if target == left and len({control, left, right}) == 3:
                block = controlled_swap_bit_ir(control, left, right)
                if gates[i : i + len(block)] == block:
                    exchanged = support.multiply(
                        values[control], support.xor(values[left], values[right])
                    )
                    values[left] = support.xor(values[left], exchanged)
                    values[right] = support.xor(values[right], exchanged)
                    i += len(block)
                    continue
        if gate.name == "h" and i + 3 < len(gates):
            target = gate.wires[0]
            if (
                gates[i + 1].name == "t"
                and gates[i + 2].name == gates[i + 3].name == "cx"
            ):
                x, y = gates[i + 2].wires[0], gates[i + 3].wires[0]
                block = and_compute_ir(x, y, target)
                if gates[i : i + len(block)] == block:
                    if values[target] != ZERO:
                        raise AssertionError(
                            f"And compute target is not proved clean at {i}"
                        )
                    values[target] = support.multiply(values[x], values[y])
                    i += len(block)
                    continue
            if gates[i + 1].name == "measure" and gates[i + 2].name == "cz":
                x, y = gates[i + 2].wires
                slot = int(gates[i + 1].metadata["classical_bit"])
                block = and_uncompute_ir(x, y, target, slot)
                if gates[i : i + len(block)] == block:
                    if values[target] != support.multiply(values[x], values[y]):
                        raise AssertionError(
                            f"And predicate was not restored before erasure at {i}"
                        )
                    values[target] = ZERO
                    events += 1
                    i += len(block)
                    continue
        if gate.metadata:
            raise AssertionError(f"Uncertified classical effect at {i}: {gate.name}")
        if gate.name in {"rx", "ry", "rz", "phase", "crx", "cry", "crz", "cphase"}:
            angle = gate.params.get("theta")
            if (
                set(gate.params) != {"theta"}
                or not isinstance(angle, Real)
                or not isfinite(angle)
            ):
                raise AssertionError(
                    "Certificate requires a finite real rotation angle"
                )
        elif gate.params:
            raise AssertionError("Certificate requires literal fixed-gate parameters")
        if gate.name == "any":
            for wire in _matrix_changed_wires(gate):
                values[wire] = support.variable(fresh)
                fresh += 1
        elif gate.name == "cswap":
            control, left, right = gate.wires
            exchanged = support.multiply(
                values[control], support.xor(values[left], values[right])
            )
            values[left] = support.xor(values[left], exchanged)
            values[right] = support.xor(values[right], exchanged)
        elif gate.name in {"x", "y"}:
            values[gate.wires[0]] = support.xor(values[gate.wires[0]], ONE)
        elif gate.name == "cx":
            control, target = gate.wires
            values[target] = support.xor(values[target], values[control])
        elif gate.name == "ccx":
            x, y, target = gate.wires
            values[target] = support.xor(
                values[target], support.multiply(values[x], values[y])
            )
        elif gate.name == "swap":
            a, b = gate.wires
            values[a], values[b] = values[b], values[a]
        elif gate.name in {"h", "ch", "rx", "ry", "crx", "cry"}:
            # The target becomes an independent support variable. Keeping every
            # old variable in other wires enlarges support and preserves soundness.
            target = gate.wires[-1]
            values[target] = support.variable(fresh)
            fresh += 1
        elif gate.name not in {
            "i",
            "z",
            "s",
            "sdg",
            "t",
            "tdg",
            "rz",
            "phase",
            "cz",
            "cphase",
            "crz",
        }:
            raise AssertionError(
                f"Unsupported unitary in certificate at {i}: {gate.name}"
            )
        i += 1
    if any(values[w] != ZERO for w in zero_wires):
        raise AssertionError("Certificate did not restore every declared clean wire")
    return events
