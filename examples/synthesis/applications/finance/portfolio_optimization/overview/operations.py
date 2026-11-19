"""Bounded construction and independent state checks for the original portfolio."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate

from ..comparisons.circuit import reference_circuit
from ..main import DEFAULT_PARAMS, main


@dataclass(frozen=True)
class StateCheck:
    parameters: tuple[float, ...]
    maximum_state_error: float
    dirty_workspace_probability: float
    resources: ResourceEstimate


def build_program(parameters: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    """Bind all six angles while preserving three assets and the slack encoding."""
    if len(parameters) != 6 or any(
        isinstance(value, bool) or not math.isfinite(value) for value in parameters
    ):
        raise ValueError("three portfolio layers require six finite radian parameters")
    return synth.synthesize(
        main,
        classical_inputs={"params": parameters},
        max_qubits=12,
        max_workspace=0,
        max_operations=4096,
        memory_limit_bytes=64 * 1024**2,
    )


def verify(parameters: tuple[float, ...] = DEFAULT_PARAMS) -> StateCheck:
    """Compare every absolute complex amplitude against separate Circuit factors."""
    program = build_program(parameters)
    expected = fq.run(reference_circuit(parameters)).statevector()
    actual = fq.run(program).statevector()
    assert isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor)
    names = ("a0", "a1", "a2", "slack")
    assert tuple(program.registers) == names
    logical = tuple(wire for name in names for wire in program.registers[name])
    width = actual.numel().bit_length() - 1
    hidden = tuple(wire for wire in range(width) if wire not in logical)
    state = actual.reshape((2,) * width).permute((*logical, *hidden)).reshape(4096, -1)
    reference = expected.flatten()
    error = float((state[:, 0] - reference).abs().max())
    dirty = float(state[:, 1:].abs().square().sum())
    torch.testing.assert_close(state[:, 0], reference, atol=3e-11, rtol=0)
    assert dirty < 1e-22
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate) and resources.n_qubits == 12
    return StateCheck(parameters, error, dirty, resources)
