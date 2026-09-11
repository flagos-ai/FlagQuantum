"""IR snapshots own their tensors and do not retain the builder's autograd graph."""

import pytest
import torch

from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode, ObservableNode
from flagquantum.core.parameters import Parameter, ParameterExpression
from flagquantum.runtime.builder_compilation import detached_ir_snapshot

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("requires_grad", [False, True])
def test_matrix_snapshot_owns_storage_and_gradient_history(requires_grad: bool) -> None:
    angle = torch.tensor(0.3, dtype=torch.float64, requires_grad=requires_grad)
    cosine, sine = angle.cos(), angle.sin()
    matrix = torch.stack((torch.stack((cosine, -sine)), torch.stack((sine, cosine))))
    ir = CircuitIR(1, (Instruction("rotation", (0,), matrix=matrix),))
    snapshot = detached_ir_snapshot(ir)
    copied = snapshot.instructions[0].matrix
    assert isinstance(copied, torch.Tensor)
    assert copied is not matrix
    assert copied.data_ptr() != matrix.data_ptr()
    assert copied.grad_fn is None
    assert copied.requires_grad is requires_grad
    assert copied.dtype == matrix.dtype
    assert copied.device == matrix.device
    torch.testing.assert_close(copied, matrix)
    assert snapshot.to_json() == ir.to_json()

    if requires_grad:
        copied.square().sum().backward()
        assert copied.grad is not None
        assert angle.grad is None

    expected = copied.detach().clone()
    with torch.no_grad():
        matrix.add_(1)
    torch.testing.assert_close(copied, expected)


@pytest.mark.parametrize("location", ["params", "coefficient", "metadata"])
def test_nested_snapshot_values_are_independent(location: str) -> None:
    source = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    value = source.sin()
    nested = {"values": [(value, "unchanged")]}
    ir = CircuitIR(
        1,
        (
            Instruction(
                "rx", (0,), params={"theta": value, "nested": nested}, metadata=nested
            ),
        ),
        observables=(ObservableNode("z", (0,), coefficient=value, metadata=nested),),
        measurements=(MeasurementNode("samples", (0,), shots=4, metadata=nested),),
        metadata=nested,
    )
    snapshot = detached_ir_snapshot(ir)
    assert snapshot.to_json() == ir.to_json()
    if location == "params":
        copied = snapshot.instructions[0].params["nested"]["values"][0][0]
    elif location == "coefficient":
        copied = snapshot.observables[0].coefficient
    else:
        copied = snapshot.metadata["values"][0][0]
    assert isinstance(copied, torch.Tensor)
    assert copied.grad_fn is None
    assert copied.requires_grad
    assert copied.data_ptr() != value.data_ptr()
    copied.backward()
    assert source.grad is None

    before = snapshot.to_json()
    with torch.no_grad():
        value.add_(1)
    nested["values"].clear()
    assert snapshot.to_json() == before


def test_symbolic_snapshot_preserves_binding_without_retaining_tensor_history() -> None:
    source = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    expression = ParameterExpression("add", (Parameter("offset"), source.sin()))
    ir = CircuitIR(1, (Instruction("rx", (0,), params={"theta": expression}),))
    snapshot = detached_ir_snapshot(ir)
    copied = snapshot.instructions[0].params["theta"]
    assert isinstance(copied, ParameterExpression)
    assert snapshot.to_json() == ir.to_json()
    assert isinstance(copied.args[1], torch.Tensor)
    assert copied.args[1].grad_fn is None
    actual = copied.bind({"offset": 0.5})
    torch.testing.assert_close(actual, expression.bind({"offset": 0.5}))
    actual.backward()
    assert source.grad is None
    torch.testing.assert_close(copied.args[1].grad, torch.ones_like(source))
