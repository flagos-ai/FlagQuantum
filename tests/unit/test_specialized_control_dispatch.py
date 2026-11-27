"""Specialization delegates before the selected owner admits physical controls."""

from typing import Any

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_feedback import (
    classical_bit_feedback_function,
)
from flagquantum.compiler.registers.control.component_adjoint_control import (
    adjoint_component_control_system,
)
from flagquantum.compiler.registers.control.component_control_selection import (
    select_component_control_system,
)
from flagquantum.compiler.registers.control.component_extension import (
    StructuredControlAdder,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    classical_unsigned_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_default_component_control import _adder

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("case", ("classical", "word", "unknown", "symbolic", "zero"))
@pytest.mark.parametrize("adjoint", (False, True))
def test_source_declaration_and_actual_fallback_are_retained(
    case: str, adjoint: bool
) -> None:
    from qualtran._infra.controlled import CtrlSpec
    from qualtran._infra.data_types import CBit, CUInt
    from qualtran.bloqs.basic_gates.cnot import CNOT
    from qualtran.bloqs.basic_gates.x_basis import XGate
    from qualtran.bloqs.mcmt.specialized_ctrl import (
        AdjointWithSpecializedCtrl,
        SpecializeOnCtrlBit,
        get_ctrl_system_1bit_cv_from_bloqs,
    )
    from qualtran.symbolics.types import Shaped

    n = sympy.Symbol("n", integer=True, positive=True)
    patterns: tuple[ControlRegister, ...] | None = None
    if case == "classical":
        control = Register("ctrl", classical_bit_type())
        patterns, spec = (ControlRegister(1, (1,)),), CtrlSpec(CBit(), 1)
    elif case == "word":
        control = Register("ctrl", classical_unsigned_register_type(2))
        patterns, spec = (ControlRegister(2, (2,)),), CtrlSpec(CUInt(2), 2)
    else:
        shape = (n,) if case == "symbolic" else (0 if case == "zero" else 2,)
        control, spec = Register("ctrl", quantum_bit_type(), shape), CtrlSpec(
            cvs=Shaped(shape)
        )
    if adjoint:
        original, _ = AdjointWithSpecializedCtrl(
            XGate(), SpecializeOnCtrlBit.BOTH
        ).get_ctrl_system(spec)
    else:
        original, _ = get_ctrl_system_1bit_cv_from_bloqs(
            XGate(),
            spec,
            current_ctrl_bit=None,
            bloq_with_ctrl=CNOT(),
            ctrl_reg_name="ctrl",
        )
    controls = (control,)
    module = ProgramRecord(
        (
            FunctionRecord(
                "main", (), registers=(control, Register("q", quantum_bit_type()))
            ),
        )
    )
    assert [
        (r.name, r.total_bits(), r.shape_symbolic)
        for r in module.functions[0].registers or ()
    ] == [(r.name, r.total_bits(), r.shape_symbolic) for r in original.signature]
    attach = _adder()
    received = []

    def fallback(
        c: tuple[Register, ...], p: tuple[ControlRegister, ...] | None
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        received.append((c, p))
        return module, attach

    def unexpected(*args: Any) -> Any:
        raise AssertionError(
            "Non-single-qubit controls must not select a specialization"
        )

    selected = (
        adjoint_component_control_system(
            ProgramRecord(()),
            controls,
            patterns,
            frozenset((0, 1)),
            unexpected,
            fallback,
            {},
        )
        if adjoint
        else select_component_control_system(controls, patterns, unexpected, fallback)
    )
    assert selected == (module, attach) and received == [(controls, patterns)]
    assert received[0][0] is controls and received[0][1] is patterns


@pytest.mark.parametrize("activation", (0, 1))
@pytest.mark.parametrize("adjoint", (False, True))
def test_classical_fallback_executes_actual_callback_and_coherent_append(
    activation: int, adjoint: bool
) -> None:
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.controlled import CtrlSpec
    from qualtran._infra.data_types import CBit
    from qualtran.bloqs.basic_gates.cnot import CNOT
    from qualtran.bloqs.basic_gates.x_basis import XGate
    from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs

    controls = (Register("ctrl", classical_bit_type()),)
    patterns = (ControlRegister(1, (activation,)),)

    def fallback(
        c: tuple[Register, ...], p: tuple[ControlRegister, ...] | None
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        assert c is controls and p is patterns
        function = classical_bit_feedback_function(
            "main",
            (Register("q", quantum_bit_type()),),
            CircuitIR(1, (Instruction("x", (0,)),)),
            (((), (bool(activation),)),),
        )
        return ProgramRecord((function,)), _adder()

    def unexpected(*args: Any) -> Any:
        raise AssertionError("Classical controls belong to feedback")

    module, attach = (
        adjoint_component_control_system(
            ProgramRecord(()),
            controls,
            patterns,
            frozenset((0, 1)),
            unexpected,
            fallback,
            {},
        )
        if adjoint
        else select_component_control_system(controls, patterns, unexpected, fallback)
    )
    builder = ComponentProgramBuilder(
        Signature((*controls, Register("q", quantum_bit_type()))),
        module.functions,
        name="caller",
    )
    (control,), (target,) = attach(
        builder, (builder.inputs["ctrl"],), {"q": builder.inputs["q"]}
    )
    caller = builder.finalize({"ctrl": control, "q": target})
    source, source_attach = get_ctrl_system_1bit_cv_from_bloqs(
        XGate(),
        CtrlSpec(CBit(), activation),
        current_ctrl_bit=None,
        bloq_with_ctrl=CNOT(),
        ctrl_reg_name="ctrl",
    )
    original, inputs = BloqBuilder.from_signature(source.signature)
    (source_control,), (source_target,) = source_attach(
        original, (inputs["ctrl"],), {"q": inputs["q"]}
    )
    composite = original.finalize(ctrl=source_control, q=source_target)
    state = torch.tensor([1 + 2j, -3j, 2 - 1j, 4 + 2j], dtype=torch.complex128)
    state /= state.norm()
    for value in (0, 1):
        _, enabled = composite.call_classically(ctrl=value, q=0)
        bound = bind_register_program(
            caller, ((), (0,)), classical_inputs=(value, None), entry="caller"
        )
        assert bound.classical_outputs == ((0,), ()) and bound.events == ()
        circuit = DynamicCircuit(2, inputs=state, dtype=torch.complex128)
        circuit.append(FieldOperation(2, bound.circuit.instructions))
        result = run_dynamic(circuit, shots=1, strategy="trajectory")
        expected = state.reshape(2, 2).flip(0).flatten() if enabled else state
        torch.testing.assert_close(
            result.final_states[0].flatten(), expected, rtol=0, atol=3e-14
        )
        assert result.classical_bits[0].tolist() == [value]


def test_unknown_single_control_is_refused_before_specialization_or_fallback() -> None:
    from qualtran._infra.controlled import CtrlSpec
    from qualtran.bloqs.basic_gates.cnot import CNOT
    from qualtran.bloqs.basic_gates.x_basis import XGate
    from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
    from qualtran.symbolics.types import Shaped

    def unexpected(*args: Any) -> Any:
        raise AssertionError("Unknown single-bit activation cannot be executed")

    with pytest.raises(ValueError, match="symbolic"):
        get_ctrl_system_1bit_cv_from_bloqs(
            XGate(),
            CtrlSpec(cvs=Shaped(())),
            current_ctrl_bit=None,
            bloq_with_ctrl=CNOT(),
            ctrl_reg_name="ctrl",
        )
    with pytest.raises(ValueError, match="concrete activation"):
        select_component_control_system(
            (Register("ctrl", quantum_bit_type()),), None, unexpected, unexpected
        )


@pytest.mark.parametrize("count", (2, 3))
def test_unknown_finite_controls_invoke_the_existing_stored_predicate(
    count: int,
) -> None:
    from qualtran._infra.controlled import CtrlSpec
    from qualtran.symbolics.types import Shaped

    from tests.unit.test_component_control import add_h, add_phase
    from tests.unit.test_component_control_selection import fixtures, native_fallback
    from tests.unit.test_component_graph import _native_trace, _source, _source_trace
    from tests.unit.test_component_structured_control import normalize_trace

    record, source, native, _ = fixtures("spectator", True)
    controls = (Register("added", quantum_bit_type(), (count,)),)
    module, attach = select_component_control_system(
        controls,
        None,
        native.get,
        native_fallback(record, {"h": add_h, "p": add_phase}),
    )
    controlled_source, source_attach = source.get_ctrl_system(
        CtrlSpec(cvs=Shaped((count,)))
    )
    signature = Signature((*controls, *(record.functions[0].registers or ())))
    builder = ComponentProgramBuilder(signature, module.functions, name="caller")
    observer = _source(signature)
    values = {
        r.name: builder.inputs[r.name] for r in record.functions[0].registers or ()
    }
    original_values = {name: observer.inputs[name] for name in values}
    added, data = attach(builder, (builder.inputs["added"],), values)
    original_added, original_data = source_attach(
        observer.builder, (observer.inputs["added"],), original_values
    )
    caller = builder.finalize(
        dict(zip((r.name for r in signature), (*added, *data), strict=True))
    )
    original = observer.builder.finalize(
        **dict(
            zip(
                (r.name for r in signature),
                (*original_added, *original_data),
                strict=True,
            )
        )
    )
    assert normalize_trace(_native_trace(caller, "caller")) == normalize_trace(
        _source_trace(observer, original)
    )
    assert tuple(values) == tuple(original_values)
    if count == 2:
        with pytest.raises(ValueError, match="bodies"):
            bind_register_program(
                caller,
                tuple((i,) for i in range(count + 2)),
                clean=(count + 2,),
                entry="caller",
            )
    else:
        bound = bind_register_program(
            caller,
            tuple((i,) for i in range(count + 2)),
            clean=tuple(range(count + 2, 2 * count + 2)),
            entry="caller",
        )
        from itertools import product

        from tests.unit.measurement_checks import forced_branch
        from tests.unit.test_component_structured_control import original_gates
        from tests.unit.test_register_subtraction import assert_source

        gates = bound.circuit.instructions
        work = tuple(range(count + 2, 2 * count + 2))
        slots = {
            g.wires[0]: g.metadata["classical_bit"]
            for g in gates
            if g.name == "measure"
        }
        expected = original_gates(
            controlled_source,
            {"ctrl": tuple(range(count)), "q": (count,), "spectator": (count + 1,)},
            work,
            slots,
        )
        assert_source(gates, expected)
        assert len([g for g in gates if g.name == "measure"]) == count
        initial = torch.zeros(1 << (2 * count + 3), dtype=torch.complex128)
        initial.reshape(1 << (count + 2), 1 << count, 2)[:, 0, :] = torch.randn(
            1 << (count + 2),
            2,
            dtype=torch.complex128,
            generator=torch.Generator().manual_seed(561),
        )
        initial /= initial.norm()
        first = None
        for history in product((0, 1), repeat=count):
            branch = forced_branch(initial, gates, history)
            torch.testing.assert_close(
                branch, forced_branch(initial, expected, history), rtol=0, atol=3e-12
            )
            if first is None:
                first = branch
            torch.testing.assert_close(branch, first, rtol=0, atol=3e-12)
            assert (
                branch.reshape(1 << (count + 2), 1 << count, 2)[:, 1:, :]
                .abs()
                .max()
                .item()
                < 3e-12
            )
            assert abs(float(branch.norm()) - 2 ** (-count / 2)) < 3e-12
