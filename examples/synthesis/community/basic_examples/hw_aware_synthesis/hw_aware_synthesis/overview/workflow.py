"""Route one native MCX intent on the original grid and star topologies."""

from dataclasses import dataclass
from typing import Literal

import torch

import flagquantum as fq
from flagquantum.compiler import (
    CouplingMap,
    ResourceEstimate,
    estimate_resources,
    route_to_topology,
)
from flagquantum.compiler.basis_conversion import convert_basis
from flagquantum.core.ir import CircuitIR
from flagquantum.synth import (
    Output,
    Program,
    QArray,
    QBit,
    X,
    allocate,
    control,
    qfunc,
    synthesize,
)

Topology = Literal["grid", "star"]
GRID_WIDTH = 18
BASIS = ("cx", "u")
GRID_EDGES = tuple(
    (node, neighbour)
    for node in range(GRID_WIDTH)
    for neighbour in (node + 1, node + 6)
    if (neighbour == node + 1 and node % 6 < 5)
    or (neighbour == node + 6 and neighbour < GRID_WIDTH)
)
STAR_EDGES = tuple((0, vertex) for vertex in range(1, GRID_WIDTH))


@qfunc
def mcx(controls: QArray[QBit], target: QBit) -> None:
    control(controls, lambda: X(target))


@qfunc
def main(size: int, controls: Output[QArray[QBit]], target: Output[QBit]) -> None:
    allocate(size, controls)
    allocate(target)
    mcx(controls, target)


@dataclass(frozen=True)
class Candidate:
    label: str
    logical_wires: tuple[int, ...]
    circuit: CircuitIR
    resources: ResourceEstimate


@dataclass(frozen=True)
class Compilation:
    topology: Topology
    candidates: tuple[Candidate, ...]
    selected: Candidate


@dataclass(frozen=True)
class Verification:
    state_error: float
    workspace_probability: float


def build_program(size: Literal[10, 12]) -> Program:
    max_width = 20 if size == 10 else GRID_WIDTH
    return synthesize(
        main,
        classical_inputs={"size": size},
        objective="depth",
        max_qubits=max_width,
        max_workspace=max_width - size - 1,
        basis=BASIS,
        memory_limit_bytes=2**25,
    )


def coupling_map(topology: Topology) -> CouplingMap:
    return CouplingMap(GRID_WIDTH, GRID_EDGES if topology == "grid" else STAR_EDGES)


def _placements(topology: Topology) -> tuple[tuple[str, tuple[int, ...]], ...]:
    natural = tuple(range(GRID_WIDTH))
    if topology == "grid":
        snake = (*range(6), *range(11, 5, -1), *range(12, 18))
        columns = tuple(row * 6 + column for column in range(6) for row in range(3))
        return ("natural", natural), ("snake", snake), ("columns", columns)
    return (
        ("natural", natural),
        ("target-hub", natural[1:13] + (0,) + natural[13:]),
        ("work-hub", natural[1:14] + (0,) + natural[14:]),
    )


def route(program: Program, topology: Topology) -> Compilation:
    """Minimize measured CX count over explicit layouts and two native routes."""
    coupling = coupling_map(topology)
    logical = (*program.registers["controls"], *program.registers["target"])
    source = fq.Circuit.from_ir(program.circuit)
    candidates = []
    for name, layout in _placements(topology):
        placed = fq.Circuit(GRID_WIDTH, dtype=torch.complex128)
        placed.compose(source, qubits=layout[: source.n_qubits])
        for strategy in ("restore_after_each_gate", "persistent_layout"):
            routed = route_to_topology(placed, coupling, strategy=strategy)
            if not routed.metadata["routing"]["mapping_restored"]:
                raise AssertionError("the selected route must restore its input layout")
            actual = convert_basis(
                routed, gates=BASIS, max_added_operations=20_000
            ).program
            if any(
                len(op.wires) > 1
                and (len(op.wires) != 2 or not coupling.has_edge(*op.wires))
                for op in actual.instructions
            ):
                raise AssertionError(
                    "route contains a gate outside the source topology"
                )
            cost = estimate_resources(actual)
            candidates.append(
                Candidate(
                    f"{name}/{strategy}",
                    tuple(layout[i] for i in logical),
                    actual,
                    cost,
                )
            )
    selected = min(
        candidates,
        key=lambda row: (
            row.resources.operation_counts["cx"],
            row.resources.depth,
            row.label,
        ),
    )
    return Compilation(topology, tuple(candidates), selected)


def verify(circuit: CircuitIR, logical: tuple[int, ...]) -> Verification:
    """Check both target values, a disabled control, and complex coherence."""
    width = circuit.n_wires
    ancillas = tuple(wire for wire in range(width) if wire not in logical)
    data_dimension = 1 << len(logical)
    cases = [0, data_dimension - 4, data_dimension - 2, data_dimension - 1]
    values = torch.tensor([1, 2j, -3, 4j], dtype=torch.complex128)
    values = values / torch.linalg.vector_norm(values)
    ordered = torch.zeros((data_dimension, 1 << len(ancillas)), dtype=torch.complex128)
    ordered[cases, 0] = values
    axes = (*logical, *ancillas)
    physical_order = tuple(axes.index(wire) for wire in range(width))
    inputs = ordered.reshape((2,) * width).permute(physical_order).flatten()
    simulation = fq.Circuit.from_ir(circuit, dtype=torch.complex128, inputs=inputs)
    final = simulation.state().reshape((2,) * width).permute(axes).reshape_as(ordered)
    expected = ordered[:, 0].clone()
    expected[-2:] = expected[-2:].flip(0)
    error = float((final[:, 0] - expected).abs().max())
    dirty = float(final[:, 1:].abs().square().sum())
    torch.testing.assert_close(final[:, 0], expected, atol=2e-11, rtol=0)
    assert dirty < 1e-22
    return Verification(error, dirty)
