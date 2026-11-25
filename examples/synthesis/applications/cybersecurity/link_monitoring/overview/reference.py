"""Independent Circuit parity phases for the source's fixed-node vertex cover."""

from collections import defaultdict
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler import ResourceEstimate

from .workflow import build_program


@dataclass(frozen=True)
class StateCheck:
    parameters: tuple[float, ...]
    maximum_state_error: float
    dirty_workspace_probability: float
    fixed_node_violation_probability: float
    resources: ResourceEstimate


def reference_circuit(parameters: tuple[float, ...]) -> fq.Circuit:
    """Expand each independent one/two-node truth table into Circuit rotations."""
    edges = (
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (2, 6),
        (2, 11),
        (3, 5),
        (4, 10),
        (5, 7),
        (7, 8),
        (7, 9),
    )
    factors = [((i,), (0.0, -1.0)) for i in range(12)]
    pair_factors = [((a, b), (-2.0, 0.0, 0.0, 0.0)) for a, b in edges]
    parities: dict[tuple[int, ...], float] = defaultdict(float)
    for bits, table in (*factors, *pair_factors):
        for mask in range(1 << len(bits)):
            coefficient = sum(
                value * (-1 if (basis & mask).bit_count() % 2 else 1)
                for basis, value in enumerate(table)
            ) / len(table)
            support = tuple(bits[j] for j in range(len(bits)) if mask & (1 << j))
            parities[support] += coefficient
    circuit = fq.Circuit(12, dtype=torch.complex128)
    circuit.x(11)
    for node in range(1, 12):
        circuit.h(11 - node)
    for support, coefficient in parities.items():
        if coefficient == 0:
            continue
        angle = parameters[0] * coefficient
        if not support:
            matrix = torch.exp(
                torch.tensor(1j * angle, dtype=torch.complex128)
            ) * torch.eye(2, dtype=torch.complex128)
            circuit.any(0, matrix=matrix)
            continue
        wires = tuple(11 - node for node in support)
        target = wires[-1]
        for control in wires[:-1]:
            circuit.cx(control, target)
        circuit.rz(target, theta=-2 * angle)
        for control in reversed(wires[:-1]):
            circuit.cx(control, target)
    for node in range(1, 12):
        circuit.rx(11 - node, theta=parameters[1])
    return circuit


def verify(parameters: tuple[float, ...]) -> StateCheck:
    program = build_program(parameters)
    expected = fq.run(reference_circuit(parameters)).statevector()
    actual = fq.run(program).statevector()
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    logical = program.registers["target"]
    width = actual.numel().bit_length() - 1
    hidden = tuple(w for w in range(width) if w not in logical)
    state = actual.reshape((2,) * width).permute((*logical, *hidden)).reshape(4096, -1)
    error = float((state[:, 0] - expected.flatten()).abs().max())
    dirty = float(state[:, 1:].abs().square().sum())
    fixed_violation = float(state[::2].abs().square().sum())
    torch.testing.assert_close(state[:, 0], expected.flatten(), atol=3e-11, rtol=0)
    assert dirty < 1e-22 and fixed_violation < 1e-22
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate) and resources.n_qubits == 12
    return StateCheck(parameters, error, dirty, fixed_violation, resources)
