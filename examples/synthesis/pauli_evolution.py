"""Native Pauli data loading and an independent Circuit reference for examples."""

import bisect
import json
import math
import random
from pathlib import Path

import torch

import flagquantum as fq

Term = tuple[float, tuple[tuple[int, str], ...]]


def load_terms(path: Path) -> tuple[int, tuple[Term, ...]]:
    """Read finite ordered Pauli data without parsing a quantum source language."""
    raw: object = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Hamiltonian data must be an object")
    width, records = raw.get("qubits"), raw.get("terms")
    if type(width) is not int or not isinstance(width, int) or not 1 <= width <= 256:
        raise ValueError("Hamiltonian width must be an integer in 1..256")
    if not isinstance(records, list) or not 1 <= len(records) <= 1024:
        raise ValueError("Hamiltonian data requires 1..1024 terms")
    terms: list[Term] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each Pauli term must be an object")
        coefficient, factors = record.get("coefficient"), record.get("support")
        if (
            type(coefficient) not in (int, float)
            or not isinstance(coefficient, (int, float))
            or abs(coefficient) > 2**53
            or not math.isfinite(coefficient)
            or not isinstance(factors, list)
        ):
            raise ValueError("Pauli terms require finite real coefficients and support")
        support: list[tuple[int, str]] = []
        seen: set[int] = set()
        for pair in factors:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Pauli support requires index/axis pairs")
            bit, axis = pair
            if (
                type(bit) is not int
                or not isinstance(bit, int)
                or not 0 <= bit < width
                or bit in seen
                or not isinstance(axis, str)
                or axis not in {"X", "Y", "Z"}
            ):
                raise ValueError(
                    "Pauli support requires distinct bounded indices and X/Y/Z"
                )
            support.append((bit, axis))
            seen.add(bit)
        terms.append((float(coefficient), tuple(support)))
    return width, tuple(terms)


def load_operator(path: Path) -> fq.Observable:
    """Construct an immutable native Observable, retaining duplicate term order."""
    _, terms = load_terms(path)
    total: fq.Observable | None = None
    axes = {"X": fq.X, "Y": fq.Y, "Z": fq.Z}
    for coefficient, support in terms:
        term = fq.I()
        for bit, axis in support:
            term = term @ axes[axis](bit)
        weighted = coefficient * term
        total = weighted if total is None else total + weighted
    assert total is not None
    return total


def reference_sequence(
    terms: tuple[Term, ...], time: float, samples: int | None, seed: int = 0
) -> tuple[tuple[int, float], ...]:
    """Invert the weight CDF independently of the compiler's choices sampler."""
    if samples is None:
        return tuple((index, time * term[0]) for index, term in enumerate(terms))
    cumulative: list[float] = []
    total = 0.0
    for coefficient, _ in terms:
        total += abs(coefficient)
        cumulative.append(total)
    if not total or not time:
        return ()
    rng = random.Random(seed)
    step = time * math.fsum(abs(coefficient) for coefficient, _ in terms) / samples
    sequence = []
    for _ in range(samples):
        index = bisect.bisect_right(cumulative, rng.random() * total, 0, len(terms) - 1)
        sequence.append((index, step if terms[index][0] > 0 else -step))
    return tuple(sequence)


def reference_circuit(
    width: int, terms: tuple[Term, ...], sequence: tuple[tuple[int, float], ...]
) -> fq.Circuit:
    """Use an opposite parity anchor and S-dagger/H basis changes for each factor."""
    circuit = fq.Circuit(width, dtype=torch.complex128)
    for index, angle in sequence:
        support = sorted(terms[index][1])
        if not support:
            circuit.x(0).phase(0, theta=-angle).x(0).phase(0, theta=-angle)
            continue
        for bit, axis in support:
            wire = width - 1 - bit
            if axis == "Y":
                circuit.gate("sdg", wire)
            if axis != "Z":
                circuit.h(wire)
        anchor = width - 1 - support[-1][0]
        for bit, _ in support[:-1]:
            circuit.cx(width - 1 - bit, anchor)
        circuit.rz(anchor, theta=2 * angle)
        for bit, _ in reversed(support[:-1]):
            circuit.cx(width - 1 - bit, anchor)
        for bit, axis in reversed(support):
            wire = width - 1 - bit
            if axis != "Z":
                circuit.h(wire)
            if axis == "Y":
                circuit.s(wire)
    return circuit
