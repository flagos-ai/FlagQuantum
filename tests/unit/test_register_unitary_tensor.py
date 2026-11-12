"""Gate-to-register tensors retain source projection, phase and Torch gradients."""

from collections.abc import Mapping
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.register_tensor import (
    register_tensor_from_unitary,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Bindings = dict[str, tuple[tuple[int, ...], ...]]


def ports(
    signature: Signature,
) -> tuple[
    Any, dict[str, Any], dict[str, Any], Bindings, Bindings, dict[tuple[str, int], int]
]:
    from qualtran import QAny, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    source = SourceSignature(
        SourceRegister(r.name, QAny(r.bitsize), r.shape, Side[r.side.name])
        for r in signature
    )
    foreign: list[dict[str, Any]] = []
    native: list[Bindings] = []
    labels: dict[tuple[str, int], int] = {}
    for direction, registers in (
        ("in", signature.lefts()),
        ("out", signature.rights()),
    ):
        original: dict[str, Any] = {}
        bound: Bindings = {}
        for register in registers:
            data = np.empty(register.shape, dtype=object)
            words = []
            for index in register.all_idxs():
                token = f"{direction}:{register.name}:{index}"
                data[index] = token
                bits = []
                for bit in range(int(register.bitsize)):
                    labels[token, bit] = 11 + 7 * len(labels)
                    bits.append(labels[token, bit])
                words.append(tuple(bits))
            original[register.name] = data if register.shape else data.item()
            bound[register.name] = tuple(words)
        foreign.append(original)
        native.append(bound)
    return source, foreign[0], foreign[1], native[0], native[1], labels


@pytest.mark.parametrize(
    "sides",
    list(product((RegisterSide.LEFT, RegisterSide.RIGHT, RegisterSide.THRU), repeat=3)),
)
@pytest.mark.parametrize("shape", [(), (1, 2), (0,)])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complete_gate_supplier_data_and_axes_match_original(
    sides: tuple[RegisterSide, ...], shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    import cirq
    from qualtran.cirq_interop._cirq_to_bloq import _my_tensors_from_gate

    signature = Signature(
        Register(name, RegisterType("opaque", width), shape if i == 0 else (), side)
        for i, (name, width, side) in enumerate(
            zip(("array", "b", "c"), (1, 1, 1), sides, strict=True)
        )
    )
    width = sum(int(r.total_bits()) for r in signature)
    dimension = 1 << width
    phases = torch.arange(dimension, dtype=torch.float64) * 0.31
    fourier = (
        torch.fft.fft(torch.eye(dimension, dtype=torch.complex128)) / dimension**0.5
    )
    unitary = (torch.exp(1j * phases)[:, None] * fourier).to(dtype)
    source, incoming, outgoing, native_in, native_out, labels = ports(signature)
    original = _my_tensors_from_gate(
        cirq.MatrixGate(unitary.to(torch.complex128).numpy(), unitary_check=False),
        source,
        incoming=incoming,
        outgoing=outgoing,
    )[0]
    data = register_tensor_from_unitary(unitary, signature)
    node = bind_register_tensor(data, signature, native_in, native_out)
    torch.testing.assert_close(
        node.tensor, torch.as_tensor(original.data, dtype=dtype), atol=0, rtol=0
    )
    assert node.labels == tuple(labels[index] for index in original.inds)
    assert data.dtype == unitary.dtype and data.device == unitary.device
    assert data.shape == tuple(
        1 << int(r.bitsize)
        for registers in (signature.rights(), signature.lefts())
        for r in registers
        for _ in r.all_idxs()
    )


@pytest.mark.parametrize(
    "sides",
    list(product((RegisterSide.LEFT, RegisterSide.RIGHT, RegisterSide.THRU), repeat=2)),
)
def test_phase_gradient_matches_original_projection_finite_difference(
    sides: tuple[RegisterSide, ...],
) -> None:
    from qualtran.simulation.tensor._tensor_data_manipulation import (
        tensor_data_from_unitary_and_signature,
    )

    signature = Signature(
        Register(name, RegisterType("uint", 1), side=side)
        for name, side in zip(("a", "b"), sides, strict=True)
    )
    source, *_ = ports(signature)
    angle = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    base = torch.fft.fft(torch.eye(4, dtype=torch.complex128)) / 2
    unitary = (
        torch.exp(1j * angle * torch.arange(1, 5, dtype=torch.float64))[:, None] * base
    )
    result = register_tensor_from_unitary(unitary, signature)
    loss = result.real.sum() + 0.29 * result.imag.sum()
    (gradient,) = torch.autograd.grad(loss, angle)

    def source_loss(theta: float) -> float:
        original = np.exp(1j * theta * np.arange(1, 5))[:, None] * base.numpy()
        data = tensor_data_from_unitary_and_signature(original, source)
        return float(data.real.sum() + 0.29 * data.imag.sum())

    expected = (source_loss(0.37 + 1e-6) - source_loss(0.37 - 1e-6)) / 2e-6
    torch.testing.assert_close(
        gradient, torch.tensor(expected, dtype=gradient.dtype), atol=2e-9, rtol=0
    )


@pytest.mark.parametrize(
    "signature",
    [
        Signature(()),
        Signature([Register("q", RegisterType("opaque", 0))]),
        Signature([Register("q", RegisterType("opaque", 1), (0,))]),
    ],
)
def test_zero_bit_projection_retains_the_explicit_scalar(signature: Signature) -> None:
    value = torch.tensor([[1j]], dtype=torch.complex128, requires_grad=True)
    data = register_tensor_from_unitary(value, signature)
    from qualtran.simulation.tensor._tensor_data_manipulation import (
        tensor_shape_from_signature,
    )

    source, *_ = ports(signature)
    assert data.shape == tensor_shape_from_signature(source) and data.item() == 1j
    (gradient,) = torch.autograd.grad(data.imag.sum(), value)
    torch.testing.assert_close(gradient, torch.tensor([[1j]], dtype=value.dtype))


def test_shared_left_right_name_uses_native_validated_projection() -> None:
    signature = Signature(
        [
            Register("q", RegisterType("bit", 1), side=RegisterSide.LEFT),
            Register("q", RegisterType("bit", 1), side=RegisterSide.RIGHT),
        ]
    )
    matrix = torch.arange(16, dtype=torch.float64).reshape(4, 4).to(torch.complex128)
    actual = register_tensor_from_unitary(matrix, signature)
    torch.testing.assert_close(
        actual, matrix.reshape(2, 2, 2, 2)[0, :, :, 0], atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "defect", ["classical", "negative_width", "negative_shape", "symbolic", "matrix"]
)
def test_invalid_unitary_boundary_fails_before_projection(defect: str) -> None:
    import sympy

    dtype = RegisterType(
        "opaque",
        (
            -1
            if defect == "negative_width"
            else sympy.Symbol("n") if defect == "symbolic" else 1
        ),
        1 if defect == "classical" else 0,
    )
    signature = Signature(
        [Register("q", dtype, (-1,) if defect == "negative_shape" else ())]
    )
    matrix = torch.eye(3 if defect == "matrix" else 2, dtype=torch.complex128)
    with pytest.raises(ValueError):
        register_tensor_from_unitary(matrix, signature)


@pytest.mark.parametrize("angle_index", range(256))
def test_actual_phase_bearing_component_provider_and_append_path(
    angle_index: int,
) -> None:
    import cirq

    import flagquantum as fq
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.compiler.registers.component_graph import iter_component_calls
    from flagquantum.compiler.registers.component_tensor_expansion import (
        expand_components_for_tensor,
    )
    from flagquantum.compiler.registers.control.component_predicate import _bit_function
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir import CircuitIR, Instruction
    from flagquantum.core.ir.program import FunctionRecord
    from flagquantum.simulation.tensor_network.component_dense import (
        contract_component_tensors,
    )
    from flagquantum.simulation.tensor_network.component_tensor_providers import (
        ComponentTensorProviders,
    )
    from flagquantum.simulation.tensor_network.models import TensorNetworkNode
    from flagquantum.simulation.unitary import get_unitary

    angle = 2 * np.pi * angle_index / 256
    ir = CircuitIR(
        2,
        (
            Instruction("h", (0,)),
            Instruction("rz", (1,), {"theta": angle}),
            Instruction("cx", (1, 0)),
        ),
    )
    bit = RegisterType("bit", 1)
    signature = Signature((Register("a", bit), Register("b", bit)))
    leaf = _bit_function("phase_leaf", tuple(signature), ir)
    builder = ComponentProgramBuilder(signature, (leaf,))
    graph = builder.finalize(builder.call_dict(leaf.name, builder.inputs))
    unitary = get_unitary(ir, dtype=torch.complex128)
    q0, q1 = cirq.LineQubit.range(2)
    source = cirq.Circuit(cirq.H(q0), cirq.rz(angle)(q1), cirq.CNOT(q1, q0))
    expected = torch.as_tensor(
        source.unitary(qubit_order=(q0, q1)), dtype=unitary.dtype
    )

    def supply(
        function: FunctionRecord,
        incoming: Mapping[str, tuple[tuple[int, ...], ...]],
        outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
    ) -> tuple[TensorNetworkNode, ...]:
        assert function.registers is not None
        child = Signature(function.registers)
        return (
            bind_register_tensor(
                register_tensor_from_unitary(unitary, child), child, incoming, outgoing
            ),
        )

    providers = ComponentTensorProviders({leaf.name: supply})
    selected, _ = expand_components_for_tensor(
        graph, providers.has_custom_tensor, full_flatten=False
    )
    order = tuple(i for i, _, _ in iter_component_calls(selected))
    actual = contract_component_tensors(selected, order, providers.supply, unitary)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)
    bound = bind_register_program(graph, ((2,), (0,)))
    state = torch.complex(
        torch.arange(1, 9, dtype=torch.float64),
        torch.arange(8, dtype=torch.float64).remainder(3) - 1,
    )
    state /= state.norm()
    circuit = fq.Circuit(3, inputs=state, dtype=state.dtype)
    circuit.append(FieldOperation(3, bound.circuit.instructions))
    p0, spectator, p2 = cirq.LineQubit.range(3)
    reference = cirq.Circuit(cirq.H(p2), cirq.rz(angle)(p0), cirq.CNOT(p0, p2))
    expected_state = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(
            reference, initial_state=state.numpy(), qubit_order=(p0, spectator, p2)
        )
        .final_state_vector
    )
    torch.testing.assert_close(
        circuit.state().flatten(), torch.as_tensor(expected_state), atol=2e-12, rtol=0
    )
    assert bound.outputs == ((2,), (0,)) and bound.events == ()
    assert tuple(g.name for g in bound.circuit.instructions) == ("h", "rz", "cx")


def test_real_native_unitary_projection_preserves_parameter_gradient() -> None:
    import flagquantum as fq
    from flagquantum.simulation.unitary import get_unitary

    angle = torch.tensor(0.71, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(1, dtype=torch.complex128).h(0).rz(0, angle).h(0)
    unitary = get_unitary(circuit, dtype=torch.complex128, device="cpu")
    signature = Signature(
        [Register("q", RegisterType("bit", 1), side=RegisterSide.RIGHT)]
    )
    data = register_tensor_from_unitary(unitary, signature)
    (gradient,) = torch.autograd.grad(data[0].real, angle)
    torch.testing.assert_close(
        gradient, -0.5 * torch.sin(angle / 2), atol=2e-12, rtol=0
    )
