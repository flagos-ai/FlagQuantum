"""Synthesize the tutorial's 14- and 50-control operations locally."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.tutorials.basics.mcx.main import my_mcx
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
    qfunc,
    synthesize,
)

BASIS = ("cx", "u")
BOSTON_PATH = Path(__file__).with_name("boston_topology.json")


@qfunc
def main(
    size: int,
    enabled: int,
    initial_target: int,
    controls: Output[QArray[QBit]],
    target: Output[QBit],
) -> None:
    allocate(size, controls)
    allocate(target)
    for bit in range(enabled):
        X(controls[bit])
    if initial_target == 1:
        X(target)
    my_mcx(controls, target)


@dataclass(frozen=True)
class Snapshot:
    n_qubits: int
    edges: tuple[tuple[int, int], ...]
    basis: tuple[str, ...]


@dataclass(frozen=True)
class BostonCircuit:
    circuit: CircuitIR
    logical: tuple[int, ...]
    physical: tuple[int, ...]
    resources: ResourceEstimate
    candidate_depths: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class AmplitudeCheck:
    error: float
    workspace_probability: float
    phase: float


def program_14(
    max_width: int, workspace: int, *, basis: tuple[str, ...] | None = None
) -> Program:
    return synthesize(
        main,
        classical_inputs={"size": 14, "enabled": 0, "initial_target": 0},
        objective="depth",
        basis=basis,
        max_qubits=max_width,
        max_workspace=workspace,
        memory_limit_bytes=128 * 1024**2,
    )


def resources(program: Program) -> ResourceEstimate:
    counted = program.report["resources"]
    assert isinstance(counted, ResourceEstimate)
    return counted


@dataclass(frozen=True)
class CXCandidate:
    label: str
    circuit: CircuitIR
    logical: tuple[int, ...]
    resources: ResourceEstimate


def _borrowed_mcx(
    circuit: fq.Circuit, controls: tuple[int, ...], target: int, dirty: tuple[int, ...]
) -> None:
    """Cancel the effect of arbitrary borrowed bits with two ladder sweeps."""
    first, second, *tail = controls
    ladder = [(dirty[i], bit, dirty[i + 1]) for i, bit in enumerate(tail[:-1])]
    for _ in range(2):
        circuit.ccx(first, second, dirty[0])
        for a, b, c in ladder:
            circuit.ccx(a, b, c)
        circuit.ccx(dirty[len(controls) - 3], tail[-1], target)
        for a, b, c in reversed(ladder):
            circuit.ccx(a, b, c)


def cx_search() -> tuple[tuple[CXCandidate, ...], CXCandidate]:
    """Compare the native and two-clean borrowed-bit schedules in the named basis."""
    program = program_14(17, 2, basis=BASIS)
    logical = (*program.registers["controls"], *program.registers["target"])
    native = CXCandidate("native", program.circuit, logical, resources(program))

    left, right = tuple(range(7)), tuple(range(7, 14))
    alternative = fq.Circuit(17, dtype=torch.complex128)
    _borrowed_mcx(alternative, left, 15, right[:5])
    _borrowed_mcx(alternative, right, 16, left[:5])
    alternative.ccx(15, 16, 14)
    _borrowed_mcx(alternative, right, 16, left[:5])
    _borrowed_mcx(alternative, left, 15, right[:5])
    lowered = convert_basis(
        alternative, gates=BASIS, max_added_operations=3_000
    ).program
    borrowed = CXCandidate(
        "two-clean borrowed bits",
        lowered,
        tuple(range(15)),
        estimate_resources(lowered),
    )
    candidates = (native, borrowed)
    return candidates, min(
        candidates,
        key=lambda row: (row.resources.operation_counts["cx"], row.resources.depth),
    )


def load_boston() -> Snapshot:
    record = json.loads(BOSTON_PATH.read_text())
    edges = tuple((a, b) for a, b in record["edges"])
    return Snapshot(int(record["n_qubits"]), edges, tuple(record["basis"]))


def topology_window(snapshot: Snapshot, start: int, width: int) -> tuple[int, ...]:
    if not 0 <= start < snapshot.n_qubits or not 1 <= width <= snapshot.n_qubits:
        raise ValueError("snapshot window is outside Boston")
    adjacent: list[list[int]] = [[] for _ in range(snapshot.n_qubits)]
    for a, b in snapshot.edges:
        adjacent[a].append(b)
        adjacent[b].append(a)
    window = [start]
    visited = {start}
    for qubit in window:
        if len(window) == width:
            break
        for neighbour in sorted(adjacent[qubit]):
            if neighbour in visited:
                continue
            visited.add(neighbour)
            window.append(neighbour)
            if len(window) == width:
                break
    if len(window) != width:
        raise ValueError("snapshot has insufficient connected qubits for the window")
    return tuple(window)


def boston_circuit() -> BostonCircuit:
    """Find the smallest emitted depth across six offline Boston routes."""
    snapshot = load_boston()
    program = program_14(21, 1, basis=BASIS)
    width = program.circuit.n_wires
    source = fq.Circuit.from_ir(program.circuit)
    logical = (*program.registers["controls"], *program.registers["target"])
    candidates = []
    for start in (0, 75, 155):
        physical = topology_window(snapshot, start, width)
        index = {vertex: i for i, vertex in enumerate(physical)}
        edges = tuple(
            (index[a], index[b]) for a, b in snapshot.edges if a in index and b in index
        )
        coupling = CouplingMap(width, edges)
        for strategy in ("restore_after_each_gate", "persistent_layout"):
            routed = route_to_topology(source, coupling, strategy=strategy)
            assert routed.metadata["routing"]["mapping_restored"]
            circuit = convert_basis(
                routed, gates=snapshot.basis, max_added_operations=120_000
            ).program
            if any(
                not coupling.has_edge(*op.wires)
                for op in circuit.instructions
                if len(op.wires) == 2
            ):
                raise AssertionError(
                    "emitted two-qubit gate is outside Boston's couplings"
                )
            cost = estimate_resources(circuit)
            label = f"{start}/{strategy}"
            candidates.append((cost.depth, label, circuit, physical, cost))
    _, _, circuit, physical, cost = min(candidates, key=lambda row: (row[0], row[1]))
    return BostonCircuit(
        circuit,
        logical,
        physical,
        cost,
        tuple((name, depth) for depth, name, *_ in candidates),
    )


def verify(
    circuit: CircuitIR, logical: tuple[int, ...], *, projective: bool = False
) -> AmplitudeCheck:
    """Compare the entire small state with a direct all-ones permutation."""
    width = circuit.n_wires
    workspace = tuple(i for i in range(width) if i not in logical)
    axes = (*logical, *workspace)
    rows = 1 << len(logical)
    seeds = torch.tensor([1, -2j, 3, 4j], dtype=torch.complex128)
    prepared = torch.zeros((rows, 1 << len(workspace)), dtype=torch.complex128)
    prepared[[0, rows - 4, rows - 2, rows - 1], 0] = seeds / torch.linalg.vector_norm(
        seeds
    )
    inputs = prepared.reshape((2,) * width).permute(
        tuple(axes.index(i) for i in range(width))
    )
    result = fq.Circuit.from_ir(
        circuit, dtype=torch.complex128, inputs=inputs.flatten()
    ).state()
    final = result.reshape((2,) * width).permute(axes).reshape_as(prepared)
    expected = prepared[:, 0].clone()
    expected[-2:] = expected[-2:].flip(0)
    phase = torch.angle(expected.vdot(final[:, 0])) if projective else torch.zeros(())
    error = float((final[:, 0] * torch.exp(-1j * phase) - expected).abs().max())
    dirty = float(final[:, 1:].abs().square().sum())
    if error >= 3e-11 or dirty >= 1e-21:
        raise AssertionError(
            (
                error,
                dirty,
                float(phase),
                expected[expected != 0],
                final[expected != 0, 0],
            )
        )
    return AmplitudeCheck(error, dirty, float(phase))


def verify_program(program: Program) -> AmplitudeCheck:
    return verify(
        program.circuit, (*program.registers["controls"], *program.registers["target"])
    )


def check_50(
    enabled: int, initial_target: int
) -> tuple[ResourceEstimate, dict[tuple[int, ...], float]]:
    """Run the full-width product-state witnesses through exact native MPS."""
    if not 0 <= enabled <= 50 or initial_target not in (0, 1):
        raise ValueError("invalid fifty-control witness")
    program = synthesize(
        main,
        classical_inputs={
            "size": 50,
            "enabled": enabled,
            "initial_target": initial_target,
        },
        objective="depth",
        max_workspace=13,
        memory_limit_bytes=2048,
    )
    target = initial_target ^ int(enabled == 50)
    check_basis_witness(program, enabled, target)
    result = fq.run(program)
    observed = dict(program.probabilities(result, registers=("target",)))
    assert abs(observed.get((target,), 0) - 1) < 1e-12
    return resources(program), observed


def check_basis_witness(program: Program, enabled: int, target: int) -> None:
    """Trace the actual full-width reversible instructions and check scratch."""
    word = 0
    for gate in program.circuit.instructions:
        if gate.name not in {"x", "cx", "ccx"}:
            raise ValueError(
                f"unexpected instruction in the basis witness: {gate.name}"
            )
        *controls, wire = gate.wires
        if all(word & (1 << control_wire) for control_wire in controls):
            word ^= 1 << wire
    # A QArray index starts at the least significant end of the named register.
    enabled_wires = program.registers["controls"][::-1][:enabled]
    (target_wire,) = program.registers["target"]
    expected = sum(1 << wire for wire in enabled_wires) | (target << target_wire)
    if word != expected:
        raise AssertionError(
            "the fifty-control circuit changed a control or left dirty workspace"
        )
