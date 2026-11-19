"""Sample both walks, postselect block zero, and compare energy and depth."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler import ResourceEstimate, estimate_resources
from flagquantum.compiler.basis_conversion import convert_basis

from .model import Model, build_model
from .quantum import LOGICAL_WIDTH, naive_circuit, optimized_program

SHOTS = 1_000
SEED = 246
PHASE_COUNT = 32
BLOCK_COUNT = 16
TARGET_BASIS = ("cx", "u")
TARGET_BUDGET = 600_000


@dataclass(frozen=True)
class Method:
    signed_phases: tuple[float, ...]
    energies: tuple[float, ...]
    block_zero_counts: tuple[int, ...]
    selected_count: int
    estimate: float
    resources: ResourceEstimate


@dataclass(frozen=True)
class Result:
    model: Model
    naive: Method
    optimized: Method
    error_tolerance: float


def decode_phase(word: int) -> float:
    return (word if word < PHASE_COUNT // 2 else word - PHASE_COUNT) / PHASE_COUNT


def measure(
    model: Model, joint: torch.Tensor, resources: ResourceEstimate, seed: int
) -> Method:
    """Draw all joint outcomes, then retain the sample rows with block zero."""
    samples = torch.multinomial(
        joint.flatten(),
        SHOTS,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(samples, minlength=BLOCK_COUNT * PHASE_COUNT).reshape(
        BLOCK_COUNT, PHASE_COUNT
    )
    retained = tuple(map(int, counts[0]))
    if not any(retained):
        raise ValueError("The sample contains no block-zero outcome")
    phases = tuple(decode_phase(word) for word in range(PHASE_COUNT))
    energies = tuple(
        model.normalization * math.cos(2 * math.pi * phase) for phase in phases
    )
    mode = max(range(PHASE_COUNT), key=retained.__getitem__)
    return Method(phases, energies, retained, sum(retained), energies[mode], resources)


def naive_method(model: Model) -> Method:
    circuit = naive_circuit(model)
    state = fq.run(circuit).statevector().reshape(2**LOGICAL_WIDTH, -1)
    if float(state[:, 1:].abs().square().sum()) > 1e-20:
        raise ValueError("The controlled-power walk must return all workspace to zero")
    joint = state[:, 0].reshape(BLOCK_COUNT, PHASE_COUNT, -1).abs().square().sum(dim=2)
    count = estimate_resources(
        convert_basis(
            circuit.to_ir(), gates=TARGET_BASIS, max_added_operations=TARGET_BUDGET
        ).program
    )
    return measure(model, joint, count, SEED)


def optimized_method(model: Model) -> Method:
    program = optimized_program(model)
    probabilities = program.probabilities(
        fq.run(program), registers=("block", "phase_var"), threshold=0
    )
    joint = torch.tensor(
        [
            [probabilities.get((block, phase), 0.0) for phase in range(PHASE_COUNT)]
            for block in range(BLOCK_COUNT)
        ],
        dtype=torch.float64,
    )
    count = estimate_resources(
        convert_basis(
            program.circuit, gates=TARGET_BASIS, max_added_operations=TARGET_BUDGET
        ).program
    )
    return measure(model, joint, count, SEED + 1)


def error_bound(model: Model) -> float:
    resolution = 1 / PHASE_COUNT
    return (
        2
        * math.pi
        * model.normalization
        * math.sin(2 * math.pi * resolution)
        * resolution
    )


def run_workflow() -> Result:
    model = build_model()
    naive = naive_method(model)
    optimized = optimized_method(model)
    tolerance = error_bound(model)
    if any(
        abs(method.estimate - model.full_ci) >= tolerance
        for method in (naive, optimized)
    ):
        raise ValueError("A measured energy exceeded the original QPE tolerance")
    return Result(model, naive, optimized, tolerance)
