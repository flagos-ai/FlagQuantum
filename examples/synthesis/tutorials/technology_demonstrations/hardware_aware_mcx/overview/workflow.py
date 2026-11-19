"""Constrained MCX workflow with Circuit preparation and independent truth oracle."""

from dataclasses import dataclass
from typing import Literal

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate, final_layout
from flagquantum.synth import (
    RZ,
    H,
    Output,
    QArray,
    QBit,
    S,
    X,
    allocate,
    control,
    hadamard_transform,
    qfunc,
)

from ..main import main

Topology = Literal["line"] | None
Scenario = Literal["zero", "coherent", "active_zero", "active_one", "near_active"]


@qfunc
def coherent(cntrl: Output[QArray[QBit]], target: Output[QBit]) -> None:
    allocate(15, cntrl)
    allocate(target)
    hadamard_transform(cntrl)
    S(cntrl[0])
    H(target)
    RZ(0.73, target)
    control(cntrl, lambda: X(target))


@qfunc
def basis_case(
    enabled: int, initial_target: int, cntrl: Output[QArray[QBit]], target: Output[QBit]
) -> None:
    allocate(15, cntrl)
    allocate(target)
    for bit in range(enabled):
        X(cntrl[bit])
    S(cntrl[0])
    if initial_target == 1:
        X(target)
    control(cntrl, lambda: X(target))


@dataclass(frozen=True)
class StateCheck:
    topology: Topology
    scenario: Scenario
    resources: ResourceEstimate
    workspace_qubits: int
    maximum_state_error: float
    dirty_workspace_probability: float


def build_program(
    topology: Topology = None, scenario: Scenario = "zero"
) -> synth.Program:
    """Keep the source's 15 controls, width ceiling18 and depth objective."""
    if topology not in (None, "line"):
        raise ValueError("use all-to-all or line connectivity")
    parameters = {"active_zero": (15, 0), "active_one": (15, 1), "near_active": (14, 0)}
    if scenario == "zero":
        entry = main
        classical = None
    elif scenario == "coherent":
        entry = coherent
        classical = None
    elif scenario in parameters:
        entry = basis_case
        enabled, initial_target = parameters[scenario]
        classical = {"enabled": enabled, "initial_target": initial_target}
    else:
        raise ValueError("unknown MCX validation scenario")
    return synth.synthesize(
        entry,
        classical_inputs=classical,
        objective="depth",
        max_qubits=18,
        max_workspace=1,
        basis=("cx", "u"),
        topology=topology,
        memory_limit_bytes=512 * 1024**2,
    )


def independent_state(scenario: Scenario) -> torch.Tensor:
    """Prepare with Circuit, then apply the MCX truth permutation directly.

    Swapping only the last two amplitudes is the independent all-ones oracle,
    avoiding a 65536-square matrix and sharing no decomposition with synth.
    """
    preparation = fq.Circuit(16, dtype=torch.complex128)
    if scenario == "coherent":
        for bit in range(15):
            preparation.h(bit)
        preparation.s(14).h(15).rz(15, theta=0.73)
    elif scenario != "zero":
        enabled = 14 if scenario == "near_active" else 15
        for bit in range(enabled):
            preparation.x(14 - bit)
        preparation.s(14)
        if scenario == "active_one":
            preparation.x(15)
    measured = fq.run(preparation).statevector()
    assert isinstance(measured, torch.Tensor)
    state = measured.flatten().clone()
    state[-2:] = state[-2:].flip(0)
    return state


def verify(
    program: synth.Program, topology: Topology, scenario: Scenario
) -> StateCheck:
    """Compare every absolute complex amplitude and all hidden working storage."""
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits <= 18
    assert set(resources.operation_counts) <= {"cx", "u3"}
    assert resources.n_qubits == 17 and program.report["workspace_qubits"] == 1
    instructions = tuple(
        g for layer in program.execution.layers for g in layer.instructions
    )
    if topology == "line":
        assert all(
            len(g.wires) == 1 or abs(g.wires[0] - g.wires[1]) == 1 for g in instructions
        )
    expected = independent_state(scenario)
    state = fq.run(program).statevector().flatten()
    logical = (*program.registers["cntrl"], *program.registers["target"])
    if topology is not None:
        # Synthesis retains the routed output layout; named readout applies it
        # automatically. Raw full-state verification must do so explicitly.
        logical = final_layout(program.circuit).map_qubits(logical)
    hidden = tuple(q for q in range(resources.n_qubits) if q not in logical)
    ordered = (
        state.reshape((2,) * resources.n_qubits)
        .permute((*logical, *hidden))
        .reshape(expected.numel(), -1)
    )
    error = float((ordered[:, 0] - expected).abs().max())
    dirty = float(ordered[:, 1:].abs().square().sum())
    torch.testing.assert_close(ordered[:, 0], expected, atol=2e-11, rtol=0)
    assert dirty < 1e-22
    return StateCheck(topology, scenario, resources, 1, error, dirty)


def run_example() -> tuple[StateCheck, ...]:
    """Run both source topologies plus nontrivial coherent and XOR witnesses."""
    checks: list[StateCheck] = []
    topologies: tuple[Topology, ...] = (None, "line")
    scenarios: tuple[Scenario, ...] = (
        "zero",
        "coherent",
        "active_zero",
        "active_one",
        "near_active",
    )
    for topology in topologies:
        for scenario in scenarios:
            program = build_program(topology, scenario)
            check = verify(program, topology, scenario)
            print(check)
            checks.append(check)
    return tuple(checks)
