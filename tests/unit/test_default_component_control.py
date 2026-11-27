"""Frozen default routing reaches native total, predicate and feedback owners."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

try:
    import numpy as np
except ModuleNotFoundError:
    pytest.importorskip("numpy")

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.controlled_programs import _controlled_atom
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from flagquantum.compiler.registers.component_control import (
    control_component_hierarchy,
    control_component_via_predicate,
)
from flagquantum.compiler.registers.component_feedback import (
    classical_bit_feedback_function,
)
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.control.component_extension import (
    StructuredControlAdder,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.control.default_component_control import (
    default_component_control_system,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    classical_unsigned_register_type,
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_control_tutorial import _source_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _adder() -> StructuredControlAdder:
    def add(
        builder: ComponentProgramBuilder,
        controls: tuple[BindingCollection, ...],
        values: dict[str, BindingCollection],
    ) -> tuple[tuple[BindingCollection, ...], tuple[BindingCollection, ...]]:
        outputs = builder.call_tuple("main", {"ctrl": controls[0], **values})
        return outputs[:1], outputs[1:]

    return add


@pytest.mark.parametrize(
    "kind,width,shape,value,expected",
    [
        ("QBit", 1, (), 1, "Controlled"),
        ("QBit", 1, (), 0, "ControlledViaAnd"),
        ("QBit", 1, (1,), 1, "ControlledViaAnd"),
        ("QBit", 1, (1, 1), 1, "ControlledViaAnd"),
        ("QAny", 1, (), 1, "ControlledViaAnd"),
        ("QUInt", 1, (), 1, "ControlledViaAnd"),
        ("QUInt", 0, (), 0, "ControlledViaAnd"),
        ("CBit", 1, (), 1, "ClassicallyControlled"),
        ("CBit", 1, (2,), 0, "ClassicallyControlled"),
        ("CUInt", 0, (), 0, "ClassicallyControlled"),
        ("CUInt", 2, (), 1, "ClassicallyControlled"),
    ],
)
def test_unchanged_source_route_and_descriptor_identity(
    kind: str, width: int, shape: tuple[int, ...], value: int, expected: str
) -> None:
    from qualtran._infra import data_types as q
    from qualtran._infra.controlled import (
        CtrlSpec,
        make_ctrl_system_with_correct_metabloq,
    )
    from qualtran.bloqs.basic_gates.x_basis import XGate

    dtype = getattr(q, kind)() if kind.endswith("Bit") else getattr(q, kind)(width)
    spec = CtrlSpec(dtype, cvs=np.full(shape, value))
    source, _ = make_ctrl_system_with_correct_metabloq(XGate(), spec)
    assert type(source).__name__ == expected
    native_type = {
        "QBit": quantum_bit_type(),
        "QAny": RegisterType("opaque", width),
        "QUInt": unsigned_register_type(width),
        "CBit": classical_bit_type(),
        "CUInt": classical_unsigned_register_type(width),
    }[kind]
    ports = (Register("caller_name", native_type, shape),)
    patterns = (ControlRegister(width, (value,) * int(np.prod(shape))),)
    module = ProgramRecord(())
    adder = _adder()
    calls = []

    def owner(name: str) -> Any:
        def build(controls: object, activation: object) -> Any:
            assert controls is ports and activation is patterns
            calls.append(name)
            return module, adder

        return build

    result = default_component_control_system(
        ports,
        patterns,
        owner("Controlled"),
        owner("ControlledViaAnd"),
        owner("ClassicallyControlled"),
    )
    assert result[0] is module and result[1] is adder
    assert calls == [expected]


def test_mixed_empty_symbolic_and_owner_failure() -> None:
    from qualtran._infra import data_types as q
    from qualtran._infra.controlled import (
        CtrlSpec,
        make_ctrl_system_with_correct_metabloq,
    )
    from qualtran.bloqs.basic_gates.x_basis import XGate
    from sympy import Symbol

    def reject(controls: object, patterns: object) -> Any:
        raise RuntimeError("selected owner admission")

    for dtypes, native in (
        ((q.QBit(), q.CBit()), (quantum_bit_type(), classical_bit_type())),
        (
            (q.QUInt(0), q.CUInt(0)),
            (unsigned_register_type(0), classical_unsigned_register_type(0)),
        ),
    ):
        with pytest.raises(NotImplementedError):
            make_ctrl_system_with_correct_metabloq(
                XGate(), CtrlSpec(dtypes, (np.array(0), np.array(0)))
            )
        with pytest.raises(NotImplementedError):
            default_component_control_system(
                tuple(Register(str(i), dtype) for i, dtype in enumerate(native)),
                (ControlRegister(1, (0,)),) * 2,
                reject,
                reject,
                reject,
            )
    with pytest.raises(AssertionError):
        CtrlSpec((), ())
    with pytest.raises(ValueError):
        default_component_control_system((), (), reject, reject, reject)
    with pytest.raises(ValueError, match="one activation"):
        default_component_control_system(
            (Register("c", quantum_bit_type()),), (), reject, reject, reject
        )
    n = Symbol("n", positive=True, integer=True)
    source, _ = make_ctrl_system_with_correct_metabloq(XGate(), CtrlSpec(q.QUInt(n), 1))
    assert type(source).__name__ == "ControlledViaAnd"
    for dtype in (quantum_bit_type(), unsigned_register_type(n), classical_bit_type()):
        with pytest.raises(RuntimeError, match="selected owner admission"):
            default_component_control_system(
                (Register("c", dtype),),
                (ControlRegister(1, (1,)),),
                reject,
                reject,
                reject,
            )


def _quantum_owner(
    controls: tuple[Register, ...],
    patterns: tuple[ControlRegister, ...] | None,
    total: bool,
) -> tuple[ProgramRecord, StructuredControlAdder]:
    bit = quantum_bit_type()
    child = bit_gate_function(
        "x", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
    )
    cx = bit_gate_function(
        "cx",
        (Register("ctrl", bit), Register("q", bit)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    builder = ComponentProgramBuilder(Signature.build(q=1), (child, cx))
    record = builder.finalize(builder.call_dict("x", builder.inputs))
    mapping = {"x": _controlled_atom("cx")}
    module = (
        control_component_hierarchy(record, mapping)
        if total
        else control_component_via_predicate(record, mapping, controls, patterns)
    )
    return module, _adder()


@pytest.mark.parametrize(
    "width,shape,values",
    [(1, (), (1,)), (1, (), (0,)), (1, (1,), (1,)), (1, (2,), (1, 0)), (2, (), (1,))],
)
def test_actual_quantum_owners_callback_gates_complex_branches_and_work(
    width: int, shape: tuple[int, ...], values: tuple[int, ...]
) -> None:
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.controlled import (
        CtrlSpec,
        make_ctrl_system_with_correct_metabloq,
    )
    from qualtran._infra.data_types import QBit, QUInt
    from qualtran._infra.registers import Signature as SourceSignature
    from qualtran.bloqs.basic_gates.x_basis import XGate
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    ports = (
        Register(
            "ctrl",
            quantum_bit_type() if width == 1 else unsigned_register_type(width),
            shape,
        ),
    )
    patterns = (ControlRegister(width, values),)
    routes = []

    def total(
        c: tuple[Register, ...], p: tuple[ControlRegister, ...] | None
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        routes.append("total")
        return _quantum_owner(c, p, True)

    def predicate(
        c: tuple[Register, ...], p: tuple[ControlRegister, ...] | None
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        routes.append("predicate")
        return _quantum_owner(c, p, False)

    module, attach = default_component_control_system(
        ports, patterns, total, predicate, predicate
    )
    expected_route = (
        "total" if width == 1 and shape == () and values == (1,) else "predicate"
    )
    assert routes == [expected_route]
    builder = ComponentProgramBuilder(
        Signature((*ports, Register("q", quantum_bit_type()))),
        module.functions,
        name="caller",
    )
    (control,), (data,) = attach(
        builder, (builder.inputs["ctrl"],), {"q": builder.inputs["q"]}
    )
    record = builder.finalize({"ctrl": control, "q": data})
    count = width * len(values)
    inputs = tuple(
        tuple(range(i * width, (i + 1) * width)) for i in range(len(values))
    ) + ((count,),)
    work = tuple(range(count + 1, 2 * count))
    bound = bind_register_program(record, inputs, clean=work, entry="caller")
    assert bound.outputs == inputs
    spec = CtrlSpec(
        QBit() if width == 1 else QUInt(width), np.array(values).reshape(shape)
    )
    source_builder, source_inputs = BloqBuilder.from_signature(
        SourceSignature.build(q=1)
    )
    source_inputs["q"] = source_builder.add(XGate(), q=source_inputs["q"])
    source_child = source_builder.finalize(**source_inputs)
    source, source_attach = make_ctrl_system_with_correct_metabloq(source_child, spec)
    source_builder, source_inputs = BloqBuilder.from_signature(source.signature)
    (source_control,), (source_data,) = source_attach(
        source_builder, (source_inputs["ctrl"],), {"q": source_inputs["q"]}
    )
    source_caller = source_builder.finalize(ctrl=source_control, q=source_data)
    measurements = {w: i for i, w in enumerate(work)}
    assert_source(
        bound.circuit.instructions,
        _source_gates(
            source_caller,
            {"ctrl": tuple(range(count)), "q": (count,)},
            work,
            measurements,
        ),
    )
    state = torch.complex(
        torch.arange(1, 2 ** (count + 2) + 1).double(),
        torch.arange(2 ** (count + 2)).double(),
    )
    state /= state.norm()
    matrix = torch.as_tensor(
        source_caller.tensor_contract().copy(), dtype=torch.complex128
    ).reshape(2 ** (count + 1), -1)
    expected = (matrix @ state.reshape(2 ** (count + 1), 2)).flatten()
    # Place a spectator after work, keeping the bound wire layout intact.
    initial = torch.zeros(2 ** (count + 2 + len(work)), dtype=torch.complex128)
    initial.reshape(2 ** (count + 1), 2 ** len(work), 2)[:, 0, :] = state.reshape(-1, 2)
    n_measurements = sum(g.name == "measure" for g in bound.circuit.instructions)
    for outcomes in product((0, 1), repeat=n_measurements):
        actual = forced_branch(initial, bound.circuit.instructions, outcomes).reshape(
            2 ** (count + 1), 2 ** len(work), 2
        )
        torch.testing.assert_close(
            actual[:, 0, :].flatten(),
            expected / 2 ** (n_measurements / 2),
            rtol=0,
            atol=3e-14,
        )
        if work:
            assert actual[:, 1:, :].abs().max().item() < 3e-14
    if not work:
        circuit = Circuit(count + 2, inputs=state, dtype=torch.complex128)
        circuit.append(FieldOperation(count + 2, bound.circuit.instructions))
        torch.testing.assert_close(
            circuit.state().flatten(), expected, rtol=0, atol=3e-14
        )
    model = get_cost_value(source, QECGatesCost())
    assert model.total_t_count() == (4 if count == 2 else 0)
    assert Counter(g.name for g in bound.circuit.instructions)["measure"] == len(work)


@pytest.mark.parametrize("values", [(1,), (0,), (1, 0)])
def test_actual_classical_owner_callback_complex_append_and_control_ownership(
    values: tuple[int, ...],
) -> None:
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.controlled import (
        CtrlSpec,
        make_ctrl_system_with_correct_metabloq,
    )
    from qualtran._infra.data_types import CBit
    from qualtran.bloqs.basic_gates.x_basis import XGate

    shape = () if len(values) == 1 else (len(values),)
    controls = (Register("ctrl", classical_bit_type(), shape),)
    patterns = (ControlRegister(1, values),)

    def feedback(
        c: tuple[Register, ...], p: tuple[ControlRegister, ...] | None
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        assert c is controls and p is patterns
        function = classical_bit_feedback_function(
            "main",
            (Register("q", quantum_bit_type()),),
            CircuitIR(1, (Instruction("x", (0,)),)),
            ((shape, tuple(bool(v) for v in values)),),
        )
        module = ProgramRecord((function,))
        return module, _adder()

    def unexpected(
        c: tuple[Register, ...], p: tuple[ControlRegister, ...] | None
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        raise AssertionError("Classical control must select the feedback owner")

    module, attach = default_component_control_system(
        controls, patterns, unexpected, unexpected, feedback
    )
    builder = ComponentProgramBuilder(
        Signature((*controls, Register("q", quantum_bit_type()))),
        module.functions,
        name="caller",
    )
    (control,), (data,) = attach(
        builder, (builder.inputs["ctrl"],), {"q": builder.inputs["q"]}
    )
    record = builder.finalize({"ctrl": control, "q": data})
    source, source_attach = make_ctrl_system_with_correct_metabloq(
        XGate(), CtrlSpec(CBit(), np.array(values).reshape(shape))
    )
    source_builder, source_inputs = BloqBuilder.from_signature(source.signature)
    (source_control,), (source_data,) = source_attach(
        source_builder, (source_inputs["ctrl"],), {"q": source_inputs["q"]}
    )
    source_caller = source_builder.finalize(ctrl=source_control, q=source_data)
    state = torch.tensor([1 + 2j, 3 - 1j, -2j, 4 + 1j], dtype=torch.complex128)
    state /= state.norm()
    for bits in product((0, 1), repeat=len(values)):
        classical = source_caller.call_classically(
            ctrl=np.asarray(bits).reshape(shape), q=0
        )
        enabled = int(classical[1])
        bound = bind_register_program(
            record,
            (*(() for _ in bits), (0,)),
            classical_inputs=(*bits, None),
            entry="caller",
        )
        assert bound.classical_outputs == tuple((i,) for i in range(len(bits))) + ((),)
        assert Counter(g.name for g in bound.circuit.instructions) == {
            "x": 1,
            "classical_xor": len(values),
        }
        circuit = DynamicCircuit(2, inputs=state, dtype=torch.complex128)
        circuit.append(FieldOperation(2, bound.circuit.instructions))
        result = run_dynamic(circuit, shots=1, strategy="trajectory")
        expected = state.reshape(2, 2).flip(0).flatten() if enabled else state
        torch.testing.assert_close(
            result.final_states[0].flatten(), expected, rtol=0, atol=3e-14
        )
        assert result.classical_bits[0].tolist() == list(bits)


def test_total_factory_replacement_keeps_existing_callback_consumer() -> None:
    ports = (Register("ctrl", quantum_bit_type()),)
    patterns = (ControlRegister(1, (1,)),)
    instructions = []
    for use_total in (True, False):

        def system(
            controls: tuple[Register, ...],
            activation: tuple[ControlRegister, ...] | None,
            total: bool = use_total,
        ) -> tuple[ProgramRecord, StructuredControlAdder]:
            return _quantum_owner(controls, activation, total)

        module, attach = default_component_control_system(
            ports, patterns, system, system, system
        )
        builder = ComponentProgramBuilder(
            Signature((*ports, Register("q", quantum_bit_type()))),
            module.functions,
            name="caller",
        )
        (control,), (data,) = attach(
            builder, (builder.inputs["ctrl"],), {"q": builder.inputs["q"]}
        )
        record = builder.finalize({"ctrl": control, "q": data})
        bound = bind_register_program(record, ((0,), (1,)), entry="caller")
        instructions.append(bound.circuit.instructions)
    assert instructions[0] == instructions[1] == (Instruction("cx", (0, 1)),)
