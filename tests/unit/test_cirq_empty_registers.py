"""Zero-wire SDK layouts retain separate native register identities."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch

try:
    import cirq
except ModuleNotFoundError:
    pytest.importorskip("cirq")

try:
    import numpy as np
except ModuleNotFoundError:
    pytest.importorskip("numpy")

from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.program_calls import _expand
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._register_import import (
    _operation_ports,
    _port_arrays,
    import_cirq_register_gate,
    import_cirq_register_optree,
)
from tests.unit.test_register_bindings import SourceContract

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
SOURCE_REGISTER: Callable[[Register], Any] = SourceContract.register


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from hashlib import sha1
    from pathlib import Path

    from qualtran._infra import gate_with_registers
    from qualtran.cirq_interop import _cirq_to_bloq

    for module, expected in (
        (gate_with_registers, "df21c8cc00ae7cac7a8bfcd4d769f76daac5515f"),
        (_cirq_to_bloq, "29c7a89a164836eb2e9af2527c2b78294b952e0e"),
    ):
        raw = Path(module.__file__).read_bytes()
        assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == expected


@dataclass(frozen=True)
class ImportedOperation:
    circuit: CircuitIR

    def append_to(self, circuit: Circuit) -> Circuit:
        return circuit.compose(self.circuit)


class DefinedGate(cirq.Gate):
    def __init__(self, definition: FunctionRecord) -> None:
        self.flagquantum_register_definition = definition

    def _num_qubits_(self) -> int:
        assert self.flagquantum_register_definition.registers is not None
        return sum(
            int(r.total_bits()) for r in self.flagquantum_register_definition.registers
        )


def ports(signature: Signature) -> dict[str, Any]:
    from flagquantum.ecosystem.cirq._register_export import (
        _named_qubits,
        _register_shape,
    )

    return {r.name: _named_qubits(cirq, r, _register_shape(r)) for r in signature}


@pytest.mark.parametrize(
    "width,shape", [(0, ()), (0, (2,)), (2, (0,)), (0, (2, 0)), (2, (2, 0))]
)
def test_source_split_merge_layouts_preserve_empty_ports(
    width: int, shape: tuple[int, ...]
) -> None:
    from qualtran._infra.gate_with_registers import merge_qubits, split_qubits

    empty = RegisterType("opaque", width)
    signature = Signature(
        (
            Register("a", empty, shape),
            Register("b", empty, shape),
            Register("q", RegisterType("bit", 1)),
        )
    )
    original = tuple(SOURCE_REGISTER(r) for r in signature)
    qregs = ports(signature)
    qubits = tuple(q for r in signature for q in qregs[r.name].flat)
    expected = split_qubits(original, qubits)
    left, right = _operation_ports(signature, qubits)
    for name in qregs:
        np.testing.assert_array_equal(left[name], expected[name])
        assert right[name].shape == expected[name].shape
    assert merge_qubits(original, **qregs) == list(qubits)
    supplied = _port_arrays(signature.lefts(), qregs)
    assert supplied["a"].shape == (*shape, width)
    program = import_cirq_register_optree(
        cirq.X(qubits[0]), signature=signature, in_quregs=qregs, out_quregs=qregs
    )
    assert verify_program_record(program) == ()
    arguments = program.functions[0].blocks[0].arguments
    count = 2 * len(tuple(signature[0].all_idxs())) + 1
    assert len(arguments) == len(set(arguments)) == count
    bound = bind_register_program(program, ((),) * (count - 1) + ((0,),))
    assert bound.outputs == ((),) * (count - 1) + ((0,),)
    assert bound.circuit.instructions == (Instruction("x", (0,)),)
    initial = torch.tensor([1 + 2j, 3 - 4j, -2 + 1j, 2j], dtype=torch.complex128)
    initial /= initial.norm()
    circuit = Circuit(2, inputs=initial, dtype=torch.complex128)
    circuit.append(ImportedOperation(bound.circuit))
    torch.testing.assert_close(
        circuit.state().flatten(),
        initial.reshape(2, 2).flip(0).flatten(),
        rtol=0,
        atol=2e-14,
    )
    circuit.append(ImportedOperation(bound.circuit))
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-14)


def definition(signature: Signature, *, name: str = "empty_ports") -> FunctionRecord:
    builder = ComponentProgramBuilder(signature, name=name)
    bit = Register("q", RegisterType("bit", 1))
    child = bit_gate_function("x", (bit,), CircuitIR(1, (Instruction("x", (0,)),)))
    builder.define(child)
    output: dict[str, object] = {}
    for reg in signature:
        if reg.name == "q":
            output["q"] = builder.call_dict("x", {"q": builder.inputs["q"]})["q"]
        elif reg.side == RegisterSide.THRU:
            output[reg.name] = builder.inputs[reg.name]
        elif reg.side == RegisterSide.LEFT:
            for word in np.asarray(builder.inputs[reg.name]).flat:
                builder.free(word)
        else:
            words = np.empty(reg.shape, dtype=object)
            for index in reg.all_idxs():
                words[index] = builder.allocate(dtype=reg.dtype)
            output[reg.name] = words if reg.shape else words[()]
    record = builder.finalize(output)
    function = record.functions[0]
    operations, returned = _expand(
        function.blocks[0], {f.name: f for f in record.functions}
    )
    block = replace(function.blocks[0], operations=operations, return_values=returned)
    result = replace(function, blocks=(block,))
    assert verify_program_record(ProgramRecord((result,))) == ()
    return result


@pytest.mark.parametrize("attached", [False, True])
@pytest.mark.parametrize("shape", [(), (2,), (0,)])
def test_declared_calls_empty_through_and_directional_ports(
    attached: bool, shape: tuple[int, ...]
) -> None:
    empty = RegisterType("opaque", 0)
    signature = Signature(
        (
            Register("left", empty, shape, RegisterSide.LEFT),
            Register("a", empty, shape),
            Register("b", empty, shape),
            Register("right", empty, shape, RegisterSide.RIGHT),
            Register("q", RegisterType("bit", 1)),
        )
    )
    function = definition(signature)
    gate = DefinedGate(function)
    wire = cirq.NamedQubit("q")
    qregs = ports(signature)
    incoming = {r.name: qregs[r.name] for r in signature.lefts()}
    outgoing = {r.name: qregs[r.name] for r in signature.rights()}
    program = import_cirq_register_optree(
        gate.on(wire),
        signature=signature,
        in_quregs=incoming,
        out_quregs=outgoing,
        gate_definitions=None if attached else {gate: function},
    )
    assert verify_program_record(program) == ()
    count = 3 * len(tuple(signature[0].all_idxs()))
    bound = bind_register_program(program, ((),) * count + ((0,),), clean=(1,))
    assert bound.outputs == ((),) * count + ((0,),)
    assert bound.circuit.instructions == (Instruction("x", (0,)),)
    assert all(not event.wires for event in bound.events)


def test_empty_type_rebinding_and_unmatched_names_use_independent_handles() -> None:
    zero = RegisterType("opaque", 0)
    child_signature = Signature(
        (
            Register("typed", RegisterType("uint", 0)),
            Register("local", zero),
            Register("q", RegisterType("bit", 1)),
        )
    )
    function = definition(child_signature)
    gate = DefinedGate(function)
    outer = Signature(
        (
            Register("typed", zero),
            Register("other", zero),
            Register("q", RegisterType("bit", 1)),
        )
    )
    qregs = ports(outer)
    program = import_cirq_register_optree(
        [gate.on(qregs["q"][0]), gate.on(qregs["q"][0])],
        signature=outer,
        in_quregs=qregs,
        out_quregs=qregs,
    )
    assert verify_program_record(program) == ()
    bound = bind_register_program(program, ((), (), (0,)), clean=(1,))
    assert bound.outputs == ((), (), (0,))
    assert bound.circuit.instructions == (
        Instruction("x", (0,)),
        Instruction("x", (0,)),
    )
    assert all(not event.wires for event in bound.events)


def test_zero_operand_gate_import_and_scalar_phase_carrier() -> None:
    signature = Signature(
        (
            Register("a", RegisterType("opaque", 0)),
            Register("b", RegisterType("opaque", 0)),
        )
    )
    module = import_cirq_register_gate(cirq.IdentityGate(0), signature=signature)
    assert module is not None
    assert len(module.functions[0].blocks[0].arguments) == 2
    bound = bind_register_program(module, ((), ()), clean=(0,))
    assert bound.outputs == ((), ()) and not bound.circuit.instructions
    qregs = ports(signature)
    phase = import_cirq_register_optree(
        cirq.global_phase_operation(1j),
        signature=signature,
        in_quregs=qregs,
        out_quregs=qregs,
    )
    bound = bind_register_program(phase, ((), ()), clean=(0,))
    initial = torch.tensor([1, 0], dtype=torch.complex128)
    circuit = Circuit(1, inputs=initial, dtype=torch.complex128)
    circuit.append(ImportedOperation(bound.circuit))
    torch.testing.assert_close(
        circuit.state().flatten(), 1j * initial, rtol=0, atol=2e-14
    )
    assert bound.outputs == ((), ())
    assert [event.wires for event in bound.events] == [(0,), (0,)]


def test_live_empty_output_collision_and_layout_failure_are_explicit() -> None:
    zero = RegisterType("opaque", 0)
    child = Signature(
        (
            Register("a", zero, side=RegisterSide.RIGHT),
            Register("q", RegisterType("bit", 1)),
        )
    )
    gate = DefinedGate(definition(child))
    signature = Signature((Register("a", zero), Register("q", RegisterType("bit", 1))))
    qregs = ports(signature)
    with pytest.raises(ValueError, match="overlaps a live empty register"):
        import_cirq_register_optree(
            gate.on(qregs["q"][0]),
            signature=signature,
            in_quregs=qregs,
            out_quregs=qregs,
        )
    malformed: Mapping[str, object] = {
        "a": np.empty((1,), dtype=object),
        "q": qregs["q"],
    }
    with pytest.raises(ValueError, match="qureg shape"):
        _port_arrays(signature.lefts(), malformed)


@pytest.mark.parametrize("shape", [(), (2,), (0,), (2, 0)])
def test_original_gate_decomposition_and_known_graph_failure(
    shape: tuple[int, ...],
) -> None:
    import qualtran

    signature = Signature(
        (
            Register("a", RegisterType("opaque", 0), shape),
            Register("b", RegisterType("opaque", 0), shape),
            Register("q", RegisterType("bit", 1)),
        )
    )
    original_signature = qualtran.Signature(SOURCE_REGISTER(r) for r in signature)

    def source_signature(self: Any) -> Any:
        return original_signature

    def decompose(self: Any, *, context: Any, **quregs: Any) -> Any:
        return [cirq.X(quregs["q"][0])]

    gate_type = type(
        "EmptyRegisterGate",
        (qualtran.GateWithRegisters,),
        {
            "signature": property(source_signature),
            "decompose_from_registers": decompose,
        },
    )
    original = gate_type()
    expected = cirq.unitary(original)
    failure = "not an available Soquet" if all(shape) else "iteration dimensions are 0"
    with pytest.raises(ValueError, match=failure):
        original.decompose_bloq()
    module = import_cirq_register_gate(original, signature=signature)
    assert module is not None
    count = 2 * len(tuple(signature[0].all_idxs()))
    bound = bind_register_program(module, ((),) * count + ((0,),))
    assert bound.circuit.instructions == (Instruction("x", (0,)),)
    actual = Circuit(1, dtype=torch.complex128)
    actual.append(ImportedOperation(bound.circuit))
    torch.testing.assert_close(
        actual.state().flatten(), torch.as_tensor(expected[:, 0]), rtol=0, atol=2e-14
    )


def test_empty_ports_preserve_measured_work_and_all_source_branches() -> None:
    from itertools import product

    from qualtran import BloqBuilder, QBit
    from qualtran import Register as OriginalRegister
    from qualtran import Signature as OriginalSignature
    from qualtran.bloqs.mcmt import And

    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_cirq_feedback_import import source_branch

    builder, inputs = BloqBuilder.from_signature(
        OriginalSignature((OriginalRegister("ctrl", QBit(), shape=(2,)),))
    )
    controls, target = builder.add(And(), ctrl=inputs["ctrl"])
    controls = builder.add(And(uncompute=True), ctrl=controls, target=target)
    composite = builder.finalize(ctrl=controls)
    first, second, spectator = cirq.NamedQubit.range(3, prefix="data")
    qregs = {"ctrl": np.array([[first], [second]], dtype=object)}
    source = composite.to_cirq_circuit(
        cirq_quregs=qregs,
        qubit_manager=cirq.GreedyQubitManager(
            prefix="register_work", maximize_reuse=True
        ),
    )
    from flagquantum.core.ir.program.register_types import quantum_bit_type

    base_signature = Signature((Register("ctrl", quantum_bit_type(), shape=(2,)),))
    base = import_cirq_register_optree(
        source, signature=base_signature, in_quregs=qregs, out_quregs=qregs
    )
    signature = Signature(
        (
            Register("a", RegisterType("opaque", 0)),
            Register("b", RegisterType("opaque", 0), shape=(2,)),
            *base_signature,
        )
    )
    with_empty = {
        "a": np.empty((0,), dtype=object),
        "b": np.empty((2, 0), dtype=object),
        **qregs,
    }
    module = import_cirq_register_optree(
        source, signature=signature, in_quregs=with_empty, out_quregs=with_empty
    )
    bound = bind_register_program(module, ((), (), (), (0,), (1,)), clean=(2,))
    baseline = bind_register_program(base, ((0,), (1,)), clean=(2,))
    assert bound.circuit.instructions == baseline.circuit.instructions
    assert bound.outputs == ((), (), (), (0,), (1,))
    primitives = (
        cirq.HPowGate,
        cirq.XPowGate,
        cirq.YPowGate,
        cirq.ZPowGate,
        cirq.CXPowGate,
        cirq.CZPowGate,
        cirq.MeasurementGate,
        cirq.ResetChannel,
    )
    source_ops = cirq.decompose(
        source,
        keep=lambda op: isinstance(op.without_classical_controls().gate, primitives),
    )
    source = cirq.Circuit(cirq.Moment((op,)) for op in source_ops)
    (work,) = source.all_qubits() - {first, second}
    from flagquantum.ecosystem.cirq._feedback_import import import_cirq_feedback

    expected_ir = import_cirq_feedback(source, qubit_order=(first, second, work)).ir
    assert bound.circuit.instructions == expected_ir.instructions
    values = np.zeros(16, dtype=np.complex128)
    clean = [index for index in range(16) if not index & 2]
    values[clean] = np.arange(1, 9) * np.exp(0.19j * np.arange(8))
    values /= np.linalg.norm(values)
    initial = torch.from_numpy(values)
    for history in product((0, 1), repeat=1):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        expected = torch.from_numpy(
            source_branch(source, (first, second, work, spectator), values, history)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=3e-12)
        assert (
            torch.count_nonzero(actual[[index for index in range(16) if index & 2]])
            == 0
        )
