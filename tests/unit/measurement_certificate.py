"""Symbolic support proof for independent measured-And erasure histories.

This proves a channel identity, not a statevector simulation. Boolean support
is represented exactly in algebraic normal form over GF(2). Rotations may
enlarge the support conservatively; a failed obligation is never ignored.
"""

from math import isfinite
from numbers import Real

from flagquantum.compiler.registers.primitives.and_compute import (
    and_compute_ir,
    and_uncompute_ir,
)
from flagquantum.core.ir import Instruction

Polynomial = frozenset[frozenset[int]]
ZERO: Polynomial = frozenset()
ONE: Polynomial = frozenset((frozenset(),))


def multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    result: set[frozenset[int]] = set()
    for a in left:
        for b in right:
            term = a | b
            if term in result:
                result.remove(term)
            else:
                result.add(term)
    return frozenset(result)


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
    if any(gate.matrix is not None for gate in gates):
        raise AssertionError("Explicit matrices require their own support proof")
    if len(set(zero_wires)) != len(zero_wires) or any(
        w < 0 or w >= n_wires for w in zero_wires
    ):
        raise ValueError("Certificate clean wires must be distinct and in range")
    values = [
        ZERO if w in zero_wires else frozenset((frozenset((w,)),))
        for w in range(n_wires)
    ]
    fresh, events, i = n_wires, 0, 0
    while i < len(gates):
        gate = gates[i]
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
                    values[target] = multiply(values[x], values[y])
                    i += len(block)
                    continue
            if gates[i + 1].name == "measure" and gates[i + 2].name == "cz":
                x, y = gates[i + 2].wires
                slot = int(gates[i + 1].metadata["classical_bit"])
                block = and_uncompute_ir(x, y, target, slot)
                if gates[i : i + len(block)] == block:
                    if values[target] != multiply(values[x], values[y]):
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
        if any(w < 0 or w >= n_wires for w in gate.wires):
            raise ValueError("Certificate operand is out of range")
        if gate.name in {"x", "y"}:
            values[gate.wires[0]] ^= ONE
        elif gate.name == "cx":
            control, target = gate.wires
            values[target] ^= values[control]
        elif gate.name == "ccx":
            x, y, target = gate.wires
            values[target] ^= multiply(values[x], values[y])
        elif gate.name == "swap":
            a, b = gate.wires
            values[a], values[b] = values[b], values[a]
        elif gate.name in {"h", "rx", "ry", "crx", "cry"}:
            # The target becomes an independent support variable. Keeping every
            # old variable in other wires enlarges support and preserves soundness.
            target = gate.wires[-1]
            values[target] = frozenset((frozenset((fresh,)),))
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
