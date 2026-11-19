"""Independent full complex Circuit references for all three source-sized models."""

from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate

from .model import BINDINGS, Model, Parameters
from .workflow import build_program


@dataclass(frozen=True)
class StateCheck:
    model: Model
    parameters: Parameters
    maximum_state_error: float
    dirty_workspace_probability: float
    resources: ResourceEstimate


def reference_circuit(model: Model, angles: Parameters) -> fq.Circuit:
    """Enumerate phase diagonals independently of the native expression tree."""
    width = {"maxcut": 3, "penalty": 9, "feasible": 5}[model]
    circuit = fq.Circuit(width, dtype=torch.complex128)
    for bit in range(width):
        circuit.h(bit)
    costs = []
    for word in range(1 << width):
        if model == "maxcut":
            bits = tuple((word >> bit) & 1 for bit in range(3))
            cost = -((bits[0] != bits[1]) + (bits[0] != bits[2]))
        else:
            a = word >> (6 if model == "penalty" else 2)
            b = (word >> 4) & 3 if model == "penalty" else word & 3
            score, mass = 3 * a + 5 * b, 2 * a + 3 * b
            cost = (
                -score + 2 * (mass + (word & 15) - 12) ** 2
                if model == "penalty"
                else (-score if mass <= 12 else 0)
            )
        costs.append(cost)
    diagonal = torch.tensor(costs, dtype=torch.float64)
    for layer in range(3):
        circuit.any(
            *range(width), matrix=torch.diag(torch.exp(1j * angles[layer] * diagonal))
        )
        for bit in range(width):
            circuit.rx(bit, angles[3 + layer])
    return circuit


def verify_program(
    model: Model, angles: Parameters, program: synth.Program
) -> StateCheck:
    actual = fq.run(program).statevector()
    expected = fq.run(reference_circuit(model, angles)).statevector()
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    amplitudes = actual.reshape(expected.numel(), -1)
    error = float((amplitudes[:, 0] - expected.flatten()).abs().max())
    dirty = float(amplitudes[:, 1:].abs().square().sum())
    if error > 2e-11 or dirty > 1e-22:
        raise AssertionError("native state disagrees with independent Circuit")
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    return StateCheck(model, angles, error, dirty, resources)


def verify_all() -> tuple[StateCheck, ...]:
    models: tuple[Model, ...] = ("maxcut", "penalty", "feasible")
    return tuple(
        verify_program(model, angles, build_program(model, angles))
        for model in models
        for angles in BINDINGS
    )
