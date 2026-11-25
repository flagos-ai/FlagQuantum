"""Compare width-first and count-first candidates at the original seven sizes."""

from dataclasses import dataclass
from time import perf_counter

import flagquantum as fq
from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.classiq_discrete_quantum_walk.overview.exact_basis import (
    exact_rules,
)
from flagquantum.compiler import estimate_resources
from flagquantum.compiler.basis_conversion import convert_basis
from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.core.ir import CircuitIR

from . import archive, compact, quantum

SIZES = tuple(range(5, 12))


@dataclass(frozen=True)
class Candidate:
    circuit: CircuitIR
    resources: ResourceEstimate
    route: str

    @property
    def cx(self) -> int:
        return self.resources.operation_counts.get("cx", 0)


@dataclass(frozen=True)
class Entry:
    position_bits: int
    width_first: Candidate
    cx_first: Candidate
    width_seconds: float
    cx_seconds: float
    candidate_count: int


def measured(circuit: CircuitIR, route: str) -> Candidate:
    resources = estimate_resources(circuit)
    if set(resources.operation_counts) - {"u3", "cx"}:
        raise AssertionError("Circle benchmark circuit escaped the same CX/U3 basis")
    return Candidate(circuit, resources, route)


def search(width: int) -> Entry:
    if type(width) is not int or width not in SIZES:
        raise ValueError("The source benchmark admits 5 to 11 position bits")
    begin = perf_counter()
    minimal = measured(compact.walk(width).to_ir(), "zero-work phase parity")
    width_seconds = perf_counter() - begin
    if minimal.resources.n_qubits != width + 1:
        raise AssertionError(
            "The width-first candidate does not meet the data-plus-coin bound"
        )
    begin = perf_counter()
    rules = exact_rules()
    candidates = [minimal]
    for cap in range(width + 2, 2 * width + 1):
        program = quantum.build(width, cap)
        circuit = convert_basis(
            program.circuit,
            gates=("cx", "u3"),
            rules=rules,
            max_added_operations=200_000,
        ).program
        match = measured(circuit, f"native qfunc cap {cap}")
        if match.resources.n_qubits > cap:
            raise AssertionError("A named-basis candidate exceeded the originating cap")
        candidates.append(match)
    picked = min(
        candidates,
        key=lambda row: (
            row.cx,
            row.resources.depth,
            row.resources.n_qubits,
            row.route,
        ),
    )
    return Entry(
        width, minimal, picked, width_seconds, perf_counter() - begin, len(candidates)
    )


def run() -> tuple[Entry, ...]:
    return tuple(search(width) for width in SIZES)


def check(candidate: Candidate, width: int) -> float:
    expected = fq.run(reference_circuit(width=width, steps=1)).statevector().flatten()
    actual = fq.run(candidate.circuit).statevector().reshape(expected.numel(), -1)
    error = float((actual[:, 0] - expected).abs().max())
    work = float(actual[:, 1:].abs().square().sum())
    if error >= 1e-10 or work >= 1e-20:
        raise AssertionError(
            f"Circle absolute-phase error {error}, workspace probability {work}"
        )
    return error


def records(
    rows: tuple[Entry, ...],
) -> tuple[tuple[archive.Record, ...], tuple[archive.Record, ...]]:
    width_first = tuple(
        archive.Record(
            row.position_bits,
            row.width_first.resources.n_qubits,
            row.width_first.cx,
            row.width_seconds,
        )
        for row in rows
    )
    cx_first = tuple(
        archive.Record(
            row.position_bits,
            row.cx_first.resources.n_qubits,
            row.cx_first.cx,
            row.cx_seconds,
        )
        for row in rows
    )
    return width_first, cx_first
