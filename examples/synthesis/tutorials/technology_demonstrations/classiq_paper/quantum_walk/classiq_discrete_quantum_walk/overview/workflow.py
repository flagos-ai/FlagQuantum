"""Bounded CX-count selection for the source six-bit periodic walk."""

import time
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.classiq_discrete_quantum_walk.main import (
    main as fourier_main,
)
from flagquantum import synth
from flagquantum.compiler.basis_conversion import BasisConversionResult, convert_basis
from flagquantum.compiler.resource_estimation import (
    ResourceEstimate,
    estimate_resources,
)

from .arithmetic_candidate import build_program as arithmetic_program
from .exact_basis import exact_rules


@dataclass(frozen=True)
class Candidate:
    """One actually converted native candidate and its same-basis resource count."""

    name: str
    program: synth.Program
    conversion: BasisConversionResult
    resources: ResourceEstimate
    synthesis_seconds: float
    conversion_seconds: float


@dataclass(frozen=True)
class Workflow:
    """Both visited candidates and a deterministic minimum-CX selection."""

    candidates: tuple[Candidate, ...]
    selected: Candidate
    elapsed_seconds: float


def fourier_program() -> synth.Program:
    """Compile the frozen Fourier identity under the source width constraint."""
    return synth.synthesize(fourier_main, max_qubits=12, max_workspace=0)


def run_workflow() -> Workflow:
    """Respect source width12 and select from two verified high-level identities."""
    started = time.perf_counter()
    candidates: list[Candidate] = []
    rules = exact_rules()
    for name, builder in (
        ("fourier", fourier_program),
        ("arithmetic", arithmetic_program),
    ):
        begin = time.perf_counter()
        program = builder()
        synthesis_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        conversion = convert_basis(
            program.circuit,
            gates=("cx", "u3"),
            max_added_operations=20_000,
            rules=rules,
        )
        conversion_seconds = time.perf_counter() - begin
        resources = estimate_resources(conversion.program, gates=("cx", "u3"))
        if resources.n_qubits > 12:
            raise ValueError("The source width ceiling is twelve physical qubits.")
        # Compare physical states under the same wire ownership; never pass a
        # converted result through a different Program's checked readout identity.
        actual = fq.run(program).statevector().flatten()
        converted = fq.run(conversion.program).statevector().flatten()
        torch.testing.assert_close(actual, converted, atol=1e-11, rtol=0)
        candidates.append(
            Candidate(
                name,
                program,
                conversion,
                resources,
                synthesis_seconds,
                conversion_seconds,
            )
        )
    selected = min(
        candidates,
        key=lambda row: (
            row.resources.operation_counts.get("cx", 0),
            row.resources.depth,
            row.resources.n_qubits,
            row.name,
        ),
    )
    return Workflow(tuple(candidates), selected, time.perf_counter() - started)
