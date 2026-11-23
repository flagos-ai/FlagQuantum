"""Explicit discard leaves use native typed selection and mixed tensor wiring."""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace

import pytest
import torch

try:
    import numpy as np
except ModuleNotFoundError:
    pytest.skip("discard reference checks require NumPy", allow_module_level=True)

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_classical import discard_word_function
from flagquantum.compiler.registers.component_discard import discard_qubit_function
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_component_root_for_tensor,
    expand_components_for_tensor,
)
from flagquantum.compiler.registers.primitives.measurement import discard_qubit_ir
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import (
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_dense import (
    contract_component_superoperator,
    contract_component_tensors_with_labels,
)
from flagquantum.simulation.tensor_network.component_tensor_providers import (
    ComponentTensorProviders,
)
from flagquantum.simulation.tensor_network.discard import discard_tensor_indices
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
)
from flagquantum.simulation.unitary import get_unitary
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_superoperator_nodes import mixed_graph, supplier

pytestmark = pytest.mark.unit
Ports = Mapping[str, tuple[tuple[int, ...], ...]]


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("quantum", (False, True))
def test_source_declarations_and_trace_labels(quantum: bool) -> None:
    from qualtran.bloqs.basic_gates import Discard, DiscardQ

    function = (discard_qubit_function if quantum else discard_word_function)("drop")
    original = DiscardQ() if quantum else Discard()
    assert verify_program_record(ProgramRecord((function,))) == ()
    source_port = tuple(original.signature)[0]
    assert function.registers is not None
    native_port = function.registers[0]
    assert (native_port.name, native_port.shape, native_port.side.name) == (
        source_port.name,
        source_port.shape,
        source_port.side.name,
    )
    assert native_port.dtype.num_qubits == source_port.dtype.num_qubits
    assert native_port.dtype.classical_bits == int(not quantum)

    providers = ComponentTensorProviders({"drop": discard_tensor_indices})
    with pytest.raises(KeyError):
        original.tensor_contract(superoperator=True)
    selected, _ = expand_component_root_for_tensor(
        ProgramRecord((function,)),
        providers.has_custom_tensor,
        "drop",
        full_flatten=False,
    )
    visited: list[TensorNetworkNode | int] = []

    def supply(
        leaf: FunctionRecord, incoming: Ports, outgoing: Ports, fresh: Callable[[], int]
    ) -> Iterable[TensorNetworkNode | int]:
        labels = providers.supply(leaf, incoming, outgoing, fresh)
        source_labels = original.my_tensors({native_port.name: 7}, {})
        assert source_labels[0].ind_tuple == (7, 0)
        result = tuple(labels)
        assert result == incoming[native_port.name][0]
        visited.extend(result)
        return result

    positions = tuple(i for i, _, _ in iter_component_calls(selected, "tensor_root"))
    like = torch.empty((), dtype=torch.complex128)
    actual = contract_component_superoperator(
        selected, positions, supply, like, "tensor_root"
    )
    torch.testing.assert_close(actual, torch.eye(2, dtype=like.dtype), rtol=0, atol=0)
    assert len(visited) == 1
    with pytest.raises(ValueError, match="discard information|classical ports"):
        contract_component_tensors_with_labels(
            selected, positions, providers.supply, like, "tensor_root"
        )
    if quantum:
        with pytest.raises(ValueError, match="requires component bodies"):
            bind_register_program(ProgramRecord((function,)), ((0,),), entry="drop")


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
@pytest.mark.parametrize("phase_on_environment", (False, True))
def test_discard_full_channel_complex_branches_and_physical_replacement(
    dtype: torch.dtype, phase_on_environment: bool
) -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran import BloqBuilder, QBit, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import CNOT, DiscardQ, Hadamard, SGate

    bit = quantum_bit_type()
    signature = Signature(
        (
            Register("ctrl", bit),
            Register("env", bit, side=RegisterSide.LEFT),
            Register("spectator", bit),
        )
    )
    phase_wire = int(phase_on_environment)
    instructions = (
        Instruction("h", (0,)),
        Instruction("s", (phase_wire,)),
        Instruction("cx", (0, 1)),
    )
    unitary_ir = CircuitIR(2, instructions)
    gate = bit_gate_function(
        "entangle", (Register("ctrl", bit), Register("target", bit)), unitary_ir
    )
    drop = discard_qubit_function("trace_environment")
    builder = ComponentProgramBuilder(signature, (gate, drop))
    outputs = builder.call_dict(
        "entangle", {"ctrl": builder.inputs["ctrl"], "target": builder.inputs["env"]}
    )
    builder.call_dict("trace_environment", {"q": outputs["target"]})
    record = builder.finalize(
        {"ctrl": outputs["ctrl"], "spectator": builder.inputs["spectator"]}
    )
    matrix = get_unitary(unitary_ir, dtype=dtype)

    def unitary(
        function: FunctionRecord,
        incoming: Ports,
        outgoing: Ports,
        fresh: Callable[[], int],
    ) -> tuple[TensorNetworkNode, ...]:
        assert function == gate
        return (
            bind_register_tensor(
                matrix, Signature(function.registers or ()), incoming, outgoing
            ),
        )

    providers = ComponentTensorProviders(
        {"entangle": unitary, "trace_environment": discard_tensor_indices}
    )
    selected, _ = expand_components_for_tensor(
        record, providers.has_custom_tensor, full_flatten=False
    )
    channel = contract_component_superoperator(
        selected,
        tuple(i for i, _, _ in iter_component_calls(selected)),
        providers.supply,
        torch.empty((), dtype=dtype),
    )
    source_sig = SourceSignature(
        (
            SourceRegister("ctrl", QBit()),
            SourceRegister("env", QBit(), side=Side.LEFT),
            SourceRegister("spectator", QBit()),
        )
    )
    source, values = BloqBuilder.from_signature(source_sig)
    values["ctrl"] = source.add(Hadamard(), q=values["ctrl"])
    port = "env" if phase_on_environment else "ctrl"
    values[port] = source.add(SGate(), q=values[port])
    values["ctrl"], values["env"] = source.add(
        CNOT(), ctrl=values["ctrl"], target=values["env"]
    )
    source.add(DiscardQ(), q=values.pop("env"))
    original = source.finalize(**values)
    tolerance = 2e-6 if dtype == torch.complex64 else 2e-14
    qubits = cirq.LineQubit.range(2)
    source_unitary = torch.as_tensor(
        cirq.unitary(
            cirq.Circuit(
                Hadamard().on(qubits[0]),
                SGate().on(qubits[phase_wire]),
                CNOT().on(*qubits),
            )
        ),
        dtype=torch.complex128,
    )
    torch.testing.assert_close(
        matrix.to(torch.complex128), source_unitary, rtol=0, atol=tolerance
    )
    np.testing.assert_allclose(
        channel.numpy(),
        original.tensor_contract(superoperator=True),
        rtol=0,
        atol=tolerance,
    )

    state = torch.complex(torch.arange(1, 9).double(), torch.arange(8).double() % 3 - 1)
    state = (state / state.norm()).to(dtype)
    density = state[:, None] * state.conj()[None, :]
    expected = torch.einsum("abij,ij->ab", channel, density)
    weight = torch.arange(1, 17, dtype=torch.float64).reshape(4, 4).to(dtype)
    trainable = state.detach().clone().requires_grad_()
    traced = torch.einsum("abij,i,j->ab", channel, trainable, trainable.conj())
    gradient = torch.autograd.grad((traced * weight).real.sum(), trainable)[0]
    prepared = (matrix @ trainable.reshape(4, 2)).reshape(2, 2, 2)
    reduced = torch.einsum("aeb,ced->abcd", prepared, prepared.conj()).reshape(4, 4)
    reference_gradient = torch.autograd.grad((reduced * weight).real.sum(), trainable)[
        0
    ]
    torch.testing.assert_close(gradient, reference_gradient, rtol=0, atol=4 * tolerance)

    clean_state = torch.kron(state.to(torch.complex128), torch.tensor([1, 0]))
    physical = (*instructions, *discard_qubit_ir(1))
    assert Counter(gate.name for gate in physical) == {
        "h": 1,
        "s": 1,
        "cx": 1,
        "reset": 1,
    }
    circuit = Circuit(4, inputs=clean_state, dtype=torch.complex128)
    circuit.append(FieldOperation(4, physical))
    result = circuit.density_matrix().reshape(2, 2, 2, 2, 2, 2, 2, 2)
    torch.testing.assert_close(
        result[:, 0, :, 0, :, 0, :, 0].reshape(4, 4),
        expected.to(torch.complex128),
        rtol=0,
        atol=4 * tolerance,
    )
    assert torch.count_nonzero(result[:, 1]) == 0
    assert torch.count_nonzero(result[..., 1]) == 0

    measurement = Instruction(
        "measure", (1,), metadata={"is_dynamic": True, "classical_bit": 0}
    )
    logical = (source_unitary @ state.to(torch.complex128).reshape(4, 2)).reshape(
        2, 2, 2
    )
    for outcome in (0, 1):
        branch = forced_branch(
            clean_state, (*instructions, measurement, *discard_qubit_ir(1)), (outcome,)
        ).reshape(2, 2, 2, 2)
        torch.testing.assert_close(
            branch[:, 0, :, 0], logical[:, outcome, :], rtol=0, atol=4 * tolerance
        )
        assert torch.count_nonzero(branch[:, 1]) == 0
        assert torch.count_nonzero(branch[..., 1]) == 0


@pytest.mark.parametrize(
    "register",
    (
        Register("q", quantum_bit_type()),
        Register("q", quantum_bit_type(), (1,), RegisterSide.LEFT),
        Register("q", RegisterType("uint", 1), side=RegisterSide.LEFT),
        Register("other", quantum_bit_type(), side=RegisterSide.LEFT),
    ),
)
def test_discard_supplier_refuses_incompatible_signatures(register: Register) -> None:
    function = replace(discard_qubit_function("drop"), registers=(register,))
    with pytest.raises(ValueError, match="scalar CBit c or QBit q"):
        discard_tensor_indices(function, {"q": ((0,),)}, {}, lambda: 1)


@pytest.mark.parametrize(
    "incoming,outgoing",
    (
        ({}, {}),
        ({"q": ()}, {}),
        ({"q": ((0, 1),)}, {}),
        ({"q": ((-1,),)}, {}),
        ({"q": ((True,),)}, {}),
        ({"q": ((0,),)}, {"q": ((1,),)}),
    ),
)
def test_discard_supplier_refuses_malformed_bindings(
    incoming: Ports, outgoing: Ports
) -> None:
    with pytest.raises(ValueError, match="discard"):
        discard_tensor_indices(
            discard_qubit_function("drop"), incoming, outgoing, lambda: 2
        )


@pytest.mark.ftqc_reference
def test_classical_discard_provider_replaces_manual_measurement_graph_supplier() -> (
    None
):
    record, source = mixed_graph(True, (2,))
    like = torch.empty((), dtype=torch.complex128)
    measured = supplier(like)
    providers = ComponentTensorProviders(
        {"first": measured, "second": measured, "drop": discard_tensor_indices}
    )
    selected, _ = expand_components_for_tensor(
        record, providers.has_custom_tensor, full_flatten=False
    )
    actual = contract_component_superoperator(
        selected,
        tuple(i for i, _, _ in iter_component_calls(selected)),
        providers.supply,
        like,
    )
    np.testing.assert_allclose(
        actual.numpy(), source.tensor_contract(superoperator=True), rtol=0, atol=2e-14
    )
    assert not providers.has_custom_tensor("projection_with_the_same_ports")
