"""Run both ansatz constructions and the original finite-shot UCCSD solve."""

import math
from dataclasses import dataclass
from itertools import combinations

import torch

import flagquantum as fq
from examples.synthesis.applications.chemistry.molecular_energy_curve.overview.model import (
    build_model,
)
from examples.synthesis.applications.chemistry.molecule_eigensolver.main import (
    build_program,
)
from flagquantum import synth
from flagquantum.algorithms import CobylaOptimizer
from flagquantum.algorithms.chemistry_domain.problem import (
    fermion_hamiltonian,
    from_native_molecule,
)
from flagquantum.algorithms.molecular import create_molecular_hamiltonian
from flagquantum.compiler import ResourceEstimate
from flagquantum.observables.fermion import FermionOperator, FermionTerm, jordan_wigner
from flagquantum.synth import CX, Output, QArray, QBit, R, X, allocate, qfunc

DISTANCE = 0.735
SHOTS = 1_000_000
ITERATIONS = 200
GEOMETRIES = {
    "H2": (("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, DISTANCE))),
    "O2": (("O", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.16))),
    "LiH": (("H", (0.0, 0.0, 0.0)), ("Li", (0.0, 0.0, 1.596))),
    "H2O": (
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.586, 0.757)),
        ("H", (0.0, 0.586, -0.757)),
    ),
    "BeH2": (
        ("Be", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.334)),
        ("H", (0.0, 0.0, -1.334)),
    ),
}


def full_hamiltonian() -> fq.Observable:
    molecule = create_molecular_hamiltonian(GEOMETRIES["H2"], basis="sto-3g")
    source = fermion_hamiltonian(from_native_molecule(molecule))
    # Move from interleaved modes to the source's all-alpha then all-beta order.
    permutation = (0, 2, 1, 3)
    terms = []
    for term in source.terms:
        groups = tuple(
            tuple(permutation[i] for i in group)
            for group in (term.creations, term.annihilations)
        )
        sign = (-1) ** sum(a > b for group in groups for a, b in combinations(group, 2))
        terms.append(
            FermionTerm(
                sign * term.coefficient,
                tuple(sorted(groups[0])),
                tuple(sorted(groups[1])),
            )
        )
    return jordan_wigner(FermionOperator(tuple(terms)), n_modes=4)


@qfunc
def hardware_ansatz(angles: list[float], state: Output[QArray[QBit]]) -> None:
    allocate(4, state)
    X(state[0])
    X(state[2])
    for layer in range(3):
        for wire in range(4):
            X(state[wire])
            R(angles[4 * layer + wire], math.pi / 2, state[wire])
        for wire in range(3):
            CX(state[wire], state[wire + 1])


def build_hardware(angles: tuple[float, ...]) -> synth.Program:
    if len(angles) != 12 or not all(map(math.isfinite, angles)):
        raise ValueError("The hardware-efficient ansatz requires twelve finite angles")
    return synth.synthesize(
        hardware_ansatz,
        classical_inputs={"angles": angles},
        max_qubits=4,
        max_workspace=0,
    )


def sample_energy(
    state: torch.Tensor,
    coefficients: tuple[float, float, float],
    generator: torch.Generator,
) -> float:
    """Draw the exact aggregate binomial counts for each Pauli measurement basis."""
    offset, x, z = coefficients
    a, b = state
    plus_x = float((a + b).abs().square()) / 2
    plus_z = float(a.abs().square())
    probabilities = torch.tensor((plus_x, plus_z), dtype=torch.float64).clamp(0, 1)
    counts = torch.binomial(
        torch.full_like(probabilities, SHOTS), probabilities, generator=generator
    )
    means = 2 * counts / SHOTS - 1
    return offset + x * float(means[0]) + z * float(means[1])


@dataclass(frozen=True)
class Result:
    full: fq.Observable
    reduced: tuple[float, float, float]
    hardware: ResourceEstimate
    tapered: ResourceEstimate
    angle: float
    sampled: float
    exact_expectation: float
    fci: float
    evaluations: int
    iterations: int
    converged: bool


def run_workflow() -> Result:
    full = full_hamiltonian()
    model = build_model(DISTANCE)
    h = model.hamiltonian.real
    coefficients = (
        float((h[0, 0] + h[1, 1]) / 2),
        float(h[0, 1]),
        float((h[0, 0] - h[1, 1]) / 2),
    )
    generator = torch.Generator().manual_seed(735)
    hardware = build_hardware(tuple(0.07 * (i + 1) for i in range(12))).report[
        "resources"
    ]
    assert isinstance(hardware, ResourceEstimate)

    def evaluate(parameters: torch.Tensor) -> torch.Tensor:
        program = build_program((float(parameters[0]),))
        state = fq.run(program).statevector().flatten()
        return torch.tensor(
            sample_energy(state, coefficients, generator), dtype=torch.float64
        )

    result = CobylaOptimizer(maxiter=ITERATIONS, rhobeg=1.0, rhoend=1e-8).minimize(
        evaluate, torch.zeros(1, dtype=torch.float64)
    )
    angle = float(result.parameters[0])
    program = build_program((angle,))
    state = fq.run(program).statevector().flatten()
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    exact_expectation = float(torch.vdot(state, model.hamiltonian @ state).real)
    return Result(
        full,
        coefficients,
        hardware,
        resources,
        angle,
        result.value,
        exact_expectation,
        model.fci,
        result.evaluations,
        result.iterations,
        result.converged,
    )
