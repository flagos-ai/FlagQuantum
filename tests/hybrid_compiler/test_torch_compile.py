from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from flagquantum.compiler._hybrid import (
    capture_source,
    specialize_and_lower,
    tensor_type,
)
from flagquantum.core.parameters import bind_parameter_value
from flagquantum.runtime.executors.statevector.hybrid_torch import (
    _statevector_expectation_op,
    hybrid_statevector_expectation,
)
from flagquantum.runtime.executors.statevector.reverse import (
    execute_torch_distributed_statevector_reverse,
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
PROGRAM = capture_source(SOURCE, (WEIGHTS, DATA))
PROGRAM32 = capture_source(
    SOURCE,
    (tensor_type("float32", (None, 4)), tensor_type("float32", (4,))),
)


def _inputs():
    weights = torch.tensor(
        [[0.2, -0.3, 0.4, -0.5]], dtype=torch.float64, requires_grad=True
    )
    data = torch.tensor(
        [0.11, -0.22, 0.33, -0.44], dtype=torch.float64, requires_grad=True
    )
    lowered = specialize_and_lower(
        PROGRAM,
        (weights, data),
        circuit_dtype="complex128",
        require_smooth_gradients=True,
    )
    parameters = [value for _, value in lowered.ordered_bindings]
    return weights, data, lowered, parameters


def _custom_value(parameters, circuit_ir_json):
    return hybrid_statevector_expectation(parameters, circuit_ir_json)


def test_functional_op_matches_existing_adjoint_and_preserves_source_views() -> None:
    weights, data, lowered, parameters = _inputs()
    circuit_json = lowered.circuit_template.to_json()
    expected_parameters = [
        parameter.detach().clone().requires_grad_(True) for parameter in parameters
    ]
    expected_circuit = lowered.circuit_template
    expected_bindings = dict(
        zip((name for name, _ in lowered.ordered_bindings), expected_parameters)
    )
    expected_circuit = replace(
        expected_circuit,
        instructions=tuple(
            replace(
                instruction,
                params=bind_parameter_value(instruction.params, expected_bindings),
            )
            for instruction in expected_circuit.instructions
        ),
    )
    expected = execute_torch_distributed_statevector_reverse(
        expected_circuit, observable_wires=(0, 3), device="cpu"
    ).value
    expected_gradients = torch.autograd.grad(expected, expected_parameters)

    actual = _custom_value(parameters, circuit_json)
    actual.backward()

    torch.testing.assert_close(actual, expected.detach(), atol=1e-10, rtol=1e-10)
    actual_gradients = tuple(
        source.grad for source in (weights, data) if source.grad is not None
    )
    flat_expected = torch.stack(expected_gradients)
    expected_weights = torch.zeros_like(weights)
    expected_data = torch.zeros_like(data)
    expected_data[:] = flat_expected[:4]
    expected_weights[0, :] = flat_expected[4:]
    torch.testing.assert_close(actual_gradients[0], expected_weights)
    torch.testing.assert_close(actual_gradients[1], expected_data)


def test_opcheck_and_gradcheck() -> None:
    _, _, lowered, parameters = _inputs()
    circuit_json = lowered.circuit_template.to_json()
    opcheck_parameters = [
        parameter.detach().clone().requires_grad_(True) for parameter in parameters
    ]

    results = torch.library.opcheck(
        _statevector_expectation_op,
        (opcheck_parameters, circuit_json),
        raise_exception=False,
    )
    assert all(result == "SUCCESS" for result in results.values()), results
    gradcheck_parameters = tuple(
        parameter.detach().clone().requires_grad_(True) for parameter in parameters
    )
    assert torch.autograd.gradcheck(
        lambda *values: _custom_value(values, circuit_json),
        gradcheck_parameters,
        eps=1e-6,
        atol=2e-7,
        rtol=2e-6,
    )


def test_torch_compile_fullgraph_forward_and_backward_match_eager() -> None:
    _, _, lowered, eager_parameters = _inputs()
    circuit_json = lowered.circuit_template.to_json()
    compiled_parameters = [
        parameter.detach().clone().requires_grad_(True)
        for parameter in eager_parameters
    ]

    eager_value = _custom_value(eager_parameters, circuit_json)
    eager_gradients = torch.autograd.grad(eager_value, eager_parameters)
    compiled = torch.compile(_custom_value, backend="aot_eager", fullgraph=True)
    compiled_value = compiled(compiled_parameters, circuit_json)
    compiled_gradients = torch.autograd.grad(compiled_value, compiled_parameters)

    repeated_parameters = [
        (parameter.detach() * 0.9).requires_grad_(True)
        for parameter in compiled_parameters
    ]
    repeated_value = compiled(repeated_parameters, circuit_json)
    repeated_gradients = torch.autograd.grad(repeated_value, repeated_parameters)
    repeated_eager_value = _custom_value(repeated_parameters, circuit_json)
    repeated_eager_gradients = torch.autograd.grad(
        repeated_eager_value, repeated_parameters
    )

    torch.testing.assert_close(compiled_value, eager_value)
    for actual, expected in zip(compiled_gradients, eager_gradients):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(repeated_value, repeated_eager_value)
    for actual, expected in zip(repeated_gradients, repeated_eager_gradients):
        torch.testing.assert_close(actual, expected)


def test_operator_fails_closed_outside_phase6_profile() -> None:
    _, _, lowered, parameters = _inputs()
    with pytest.raises(ValueError, match="expected"):
        _custom_value(parameters[:-1], lowered.circuit_template.to_json())
    invalid = [*parameters]
    invalid[0] = invalid[0].reshape(1)
    with pytest.raises(ValueError, match="scalar tensor"):
        _custom_value(invalid, lowered.circuit_template.to_json())


def test_operator_supports_float32_complex64_profile() -> None:
    weights = torch.tensor(
        [[0.2, -0.3, 0.4, -0.5]], dtype=torch.float32, requires_grad=True
    )
    data = torch.tensor(
        [0.11, -0.22, 0.33, -0.44], dtype=torch.float32, requires_grad=True
    )
    lowered = specialize_and_lower(
        PROGRAM32,
        (weights, data),
        circuit_dtype="complex64",
        require_smooth_gradients=True,
    )

    value = _custom_value(
        [parameter for _, parameter in lowered.ordered_bindings],
        lowered.circuit_template.to_json(),
    )
    value.backward()

    assert value.dtype == torch.float32
    assert weights.grad is not None and torch.isfinite(weights.grad).all()
    assert data.grad is not None and torch.isfinite(data.grad).all()
