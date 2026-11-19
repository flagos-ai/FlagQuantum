"""Keep the runnable source starter and an explicit local reference solution."""

from dataclasses import dataclass
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum.compiler import ResourceEstimate
from flagquantum.core.ir import CircuitIR
from flagquantum.synth import (
    Output,
    Program,
    QArray,
    QBit,
    allocate,
    hadamard_transform,
    prepare_state,
    qfunc,
    synthesize,
)

STAGE_WIDTHS = {1: 8, 2: 4}
STAGE_CAPS = {1: 18, 2: 127}
BASIS = ("cx", "u")


@dataclass(frozen=True)
class Comparison:
    grid: tuple[float, ...]
    starter: tuple[float, ...]
    reference: tuple[float, ...]
    gaussian: tuple[float, ...]
    starter_mse: float
    reference_mse: float
    absolute_starter_error: float
    absolute_reference_error: float
    starter_sample: float


@dataclass(frozen=True)
class Stage:
    starter: Program
    reference: Program


@qfunc
def starter(width: int, state: Output[QArray[QBit]]) -> None:
    allocate(width, state)
    hadamard_transform(state)


@qfunc
def reference_answer(probabilities: list[float], state: Output[QArray[QBit]]) -> None:
    prepare_state(probabilities, 0, state)


def signed_grid(width: int) -> torch.Tensor:
    high_bit = 1 << (width - 1)
    scale = 2 ** (width - 2)
    return torch.tensor(
        [
            (word if word < high_bit else word - 2 * high_bit) / scale
            for word in range(2 * high_bit)
        ],
        dtype=torch.float64,
    )


def gaussian(width: int) -> torch.Tensor:
    values = torch.exp(-signed_grid(width).square())
    return values / values.sum()


def build(stage_number: int) -> Stage:
    if stage_number not in STAGE_WIDTHS:
        raise ValueError("choose challenge stage 1 or 2")
    width, cap = STAGE_WIDTHS[stage_number], STAGE_CAPS[stage_number]
    start = synthesize(
        starter,
        classical_inputs={"width": width},
        max_qubits=cap,
        max_workspace=0,
        basis=BASIS,
        memory_limit_bytes=2**20,
    )
    reference = synthesize(
        reference_answer,
        classical_inputs={"probabilities": gaussian(width).tolist()},
        max_qubits=cap,
        max_workspace=0,
        basis=BASIS,
        memory_limit_bytes=2**20,
    )
    return Stage(start, reference)


def resources(program: Program) -> tuple[int, int, int, int]:
    estimate = program.report["resources"]
    assert isinstance(estimate, ResourceEstimate)
    return (
        estimate.n_qubits,
        estimate.n_operations,
        estimate.depth,
        estimate.operation_counts.get("cx", 0),
    )


def compare(stage: Stage) -> Comparison:
    if any(p.circuit.n_wires != 8 for p in (stage.starter, stage.reference)):
        raise ValueError("the source executes only the eight-qubit first stage")
    expected = gaussian(8)
    trial = fq.run(stage.starter).statevector().flatten()
    solved = fq.run(stage.reference).statevector().flatten()
    error_start = float((trial - 1 / 16).abs().max())
    error_ref = float((solved - expected.sqrt()).abs().max())
    assert error_start < 3e-12 and error_ref < 3e-12
    first = trial.abs().square()
    second = solved.abs().square()
    sector = signed_grid(8)
    order = sector.argsort()
    shot = torch.multinomial(first, 1, generator=torch.Generator().manual_seed(13))
    return Comparison(
        tuple(sector[order].tolist()),
        tuple(first[order].tolist()),
        tuple(second[order].tolist()),
        tuple(expected[order].tolist()),
        float((first - expected).square().mean()),
        float((second - expected).square().mean()),
        error_start,
        error_ref,
        float(sector[int(shot.item())]),
    )


def save_program(program: Program, folder: Path, stage: int, team: str | None) -> Path:
    if team is None:
        raise ValueError("set a team name before enabling native export")
    safe = team.strip().replace(" ", "_")
    if not safe or not safe.replace("_", "").isalnum():
        raise ValueError(
            "team name must contain letters, numbers, spaces or underscores"
        )
    folder.mkdir(parents=True, exist_ok=True)
    artifact = (
        folder / f"{safe}__stage_{stage}__resolution_{program.circuit.n_wires}.json"
    )
    artifact.write_text(program.circuit.to_json() + "\n")
    assert (
        CircuitIR.from_json(artifact.read_text()).content_hash
        == program.circuit.content_hash
    )
    return artifact
