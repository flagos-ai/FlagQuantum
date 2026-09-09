from __future__ import annotations

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._hybrid import (
    SpecializationError,
    capture_source,
    specialize_and_lower,
    tensor_type,
)

pytestmark = pytest.mark.integration

WEIGHTS = tensor_type("float64", (None, 4))
DATA = tensor_type("float64", (4,))

SOURCE = """
def cost(weights, data):
    qp.AngleEmbedding(data, wires=range(4))
    for row in weights:
        for wire, parameter in enumerate(row):
            if parameter > 0:
                qp.RX(parameter, wires=wire)
            elif parameter < 0:
                qp.RY(parameter, wires=wire)
        for wire in range(4):
            qp.CNOT(wires=[wire, jnp.mod(wire + 1, 4)])
    return qp.expval(qp.PauliZ(0) + qp.PauliZ(3))
"""


def _hybrid_forward(weights: torch.Tensor, data: torch.Tensor) -> fq.ExecutionResult:
    program = capture_source(SOURCE, (WEIGHTS, DATA))
    bound = specialize_and_lower(program, (weights, data)).bind()
    return fq.run(bound)


def _reference_forward(weights: torch.Tensor, data: torch.Tensor) -> fq.ExecutionResult:
    circuit = fq.Circuit(4)
    for wire, parameter in enumerate(data):
        circuit.rx(wire, theta=parameter)
    for row in weights:
        for wire, parameter in enumerate(row):
            if bool(parameter > 0):
                circuit.rx(wire, theta=parameter)
            elif bool(parameter < 0):
                circuit.ry(wire, theta=parameter)
        for wire in range(4):
            circuit.cx(wire, (wire + 1) % 4)
    return fq.run(circuit, outputs=fq.expectation(fq.Z(0) + fq.Z(3)))


@pytest.mark.parametrize(
    "values",
    (
        (0.2, 0.3, 0.4, 0.5),
        (-0.2, -0.3, -0.4, -0.5),
        (0.0, 0.0, 0.0, 0.0),
        (0.2, -0.3, 0.0, 0.5),
    ),
)
def test_local_cpu_forward_matches_explicit_reference(
    values: tuple[float, ...],
) -> None:
    weights = torch.tensor([values], dtype=torch.float64)
    data = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float64)

    actual = _hybrid_forward(weights, data)
    expected = _reference_forward(weights, data)

    torch.testing.assert_close(actual.expectation(), expected.expectation())
    assert actual.expectation().shape == (1,)
    assert actual.summary()["distribution_semantics"] == "single_device_fast_path"
    assert actual.summary()["execution_path"] == "local_statevector"


def test_compiler_input_error_remains_owned_by_specialization() -> None:
    program = capture_source(SOURCE, (WEIGHTS, DATA))

    with pytest.raises(SpecializationError, match="input.shape"):
        specialize_and_lower(
            program,
            (
                torch.zeros((1, 4), dtype=torch.float64),
                torch.zeros(5, dtype=torch.float64),
            ),
        )
