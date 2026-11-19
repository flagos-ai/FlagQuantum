"""Unknown activations retain source connections and require explicit resolution."""

from dataclasses import replace
from itertools import product
from typing import Any

import pytest
import sympy
import torch

from flagquantum.compiler._component_trace.program_cost_query import (
    query_program_qec_gate_cost,
)
from flagquantum.compiler._component_trace.program_logical_width import (
    query_program_logical_width,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    BookkeepingCall,
    ConjunctionCall,
    ResourceCall,
)
from flagquantum.compiler.registers.component_control import (
    control_component_via_predicate,
)
from flagquantum.compiler.registers.component_graph import _namespace_helpers
from flagquantum.compiler.registers.control.component_predicate import (
    control_predicate_program,
)
from flagquantum.compiler.registers.control.conjunction_program import (
    conjunction_program,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import unsigned_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_control import add_h, add_phase, fixture_program
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_structured_control import normalize_trace

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width,shape", [(2, ()), (3, ()), (1, (2, 2))])
@pytest.mark.parametrize("uncompute", [False, True])
def test_unknown_activation_connections_resources_and_physical_refusal(
    width: int, shape: tuple[int, ...], uncompute: bool
) -> None:
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.symbolics import Shaped

    original = CtrlSpecAnd(CtrlSpec(QUInt(width), cvs=Shaped(shape)))
    source = original.adjoint() if uncompute else original
    controls = (Register("ctrl_0", unsigned_register_type(width), shape),)
    record = control_predicate_program(controls, None, uncompute=uncompute)
    signature = Signature(record.functions[0].registers or ())
    assert verify_program_record(record) == ()
    assert _native_trace(record, "predicate") == _source_trace(
        _source(signature), source.decompose_bloq()
    )
    declarations = [f for f in record.functions if not f.blocks]
    count = int(original.n_ctrl_qubits)
    assert len(declarations) == (1 if count == 2 else 0)
    conjunction = next(f for f in record.functions if f.name == "predicate_conjunction")
    # Resource semantics are supplied explicitly, independently of callee spelling.
    keys: dict[str, ResourceCall] = {
        f.name: BookkeepingCall("partition", count) for f in record.functions
    }
    keys[conjunction.name] = ConjunctionCall(
        tuple(sympy.symbols("cv1 cv2")) if count == 2 else count, uncompute
    )
    # Stop at the actual source conjunction boundary for model comparison.
    from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost

    native = query_program_qec_gate_cost(
        record,
        keys,
        entry="predicate",
        static_costs={conjunction.name: query_qec_gate_cost(keys[conjunction.name])},
    )
    model = get_cost_value(source, QECGatesCost())
    for category in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert getattr(native, category) == getattr(model, category)
    assert query_program_logical_width(record, "predicate") == get_cost_value(
        source, QubitCount()
    )
    if count == 2:
        with pytest.raises(ValueError, match="requires component bodies"):
            bind_register_program(record, (tuple(range(count)),), entry="predicate")
    else:
        concrete = control_predicate_program(
            controls,
            (ControlRegister(width, ((1 << width) - 1,) * (count // width)),),
            uncompute=uncompute,
        )
        assert record == concrete


def resolve_conjunctions(
    record: ProgramRecord, values: tuple[int, ...]
) -> ProgramRecord:
    """Replace declarations using the existing finite emitter and its typed ports."""
    functions = []
    for function in record.functions:
        if function.blocks:
            functions.append(function)
            continue
        uncompute = all(
            r.name != "target" for r in Signature(function.registers or ()).rights()
        )
        replacement = _namespace_helpers(
            conjunction_program(values, name=function.name, uncompute=uncompute),
            function.name,
        )
        assert replacement.functions[0].registers == function.registers
        functions.extend(replacement.functions)
    return replace(record, functions=tuple(functions))


@pytest.mark.parametrize("count", [2])
def test_unknown_controlled_graph_resolves_without_changing_consumers(
    count: int,
) -> None:
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
    from qualtran.symbolics import Shaped

    record, positive = fixture_program(1)
    controls = (Register("ctrl", unsigned_register_type(count)),)
    unknown = control_component_via_predicate(
        record, {"h": add_h, "p": add_phase}, controls, None
    )
    source = ControlledViaAnd(positive.subbloq, CtrlSpec(QUInt(count), cvs=Shaped(())))
    assert normalize_trace(_native_trace(unknown)) == normalize_trace(
        _source_trace(_source(Signature([])), source.decompose_bloq())
    )
    assert len([f for f in unknown.functions if not f.blocks]) == 2
    inputs = (tuple(range(count)), (count,), (count + 1,))
    work = tuple(range(count + 2, 2 * count + 2))
    with pytest.raises(ValueError, match="requires component bodies"):
        bind_register_program(unknown, inputs, clean=work)
    for values in product((0, 1), repeat=count):
        activation = int("".join(map(str, values)), 2)
        concrete = control_component_via_predicate(
            record,
            {"h": add_h, "p": add_phase},
            controls,
            (ControlRegister(count, (activation,)),),
        )
        resolved = resolve_conjunctions(unknown, values)
        assert verify_program_record(resolved) == ()
        bound = bind_register_program(resolved, inputs, clean=work)
        expected = bind_register_program(concrete, inputs, clean=work)
        assert bound == expected
        state = torch.zeros(1 << (2 * count + 2), dtype=torch.complex128)
        for index in range(1 << (count + 2)):
            state[index << count] = torch.tensor(
                complex(index + 1, 2 - index) / (1 << (count + 2)),
                dtype=torch.complex128,
            )
        for history in product((0, 1), repeat=count):
            actual = forced_branch(state, bound.circuit.instructions, history)
            reference = forced_branch(state, expected.circuit.instructions, history)
            torch.testing.assert_close(actual, reference, atol=3e-12, rtol=0)
            assert torch.count_nonzero(actual.reshape(-1, 1 << count)[:, 1:]) == 0


def test_source_unknown_multiand_resolves_its_own_decomposition_to_positive() -> None:
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.mcmt import MultiAnd
    from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd
    from qualtran.symbolics import HasLength, Shaped

    source = CtrlSpecAnd(CtrlSpec(QUInt(3), cvs=Shaped(())))
    child = next(
        instance.bloq
        for instance, _, _ in source.decompose_bloq().iter_bloqsoqs()
        if isinstance(instance.bloq, MultiAnd)
    )
    assert source._flat_cvs == HasLength(3)
    assert child.cvs == (1, 1, 1)
    assert child.concrete_cvs == (1, 1, 1)
    assert child.decompose_bloq() == MultiAnd((1, 1, 1)).decompose_bloq()


def test_unknown_wide_controlled_graph_matches_source_positive_conversion() -> None:
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
    from qualtran.symbolics import Shaped

    record, positive = fixture_program(1)
    controls = (Register("ctrl", unsigned_register_type(3)),)
    unknown = control_component_via_predicate(
        record, {"h": add_h, "p": add_phase}, controls, None
    )
    concrete = control_component_via_predicate(
        record, {"h": add_h, "p": add_phase}, controls, (ControlRegister(3, (7,)),)
    )
    source = ControlledViaAnd(positive.subbloq, CtrlSpec(QUInt(3), cvs=Shaped(())))
    assert unknown == concrete
    assert normalize_trace(_native_trace(unknown)) == normalize_trace(
        _source_trace(_source(Signature([])), source.decompose_bloq())
    )
    assert all(f.blocks for f in unknown.functions)


@pytest.mark.parametrize("old_width,new_width", [(1, 1), (2, 1), (1, 2)])
@pytest.mark.parametrize(
    "unknown_old,unknown_new", [(True, False), (False, True), (True, True)]
)
def test_unknown_extension_invokes_actual_source_compatible_adder(
    old_width: int, new_width: int, unknown_old: bool, unknown_new: bool
) -> None:
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
    from qualtran.symbolics import Shaped

    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.compiler.registers.control.component_extension import (
        extend_component_control_system,
    )

    record, positive = fixture_program(1)
    old_spec = CtrlSpec(QUInt(old_width), cvs=Shaped(()) if unknown_old else 0)
    new_spec = CtrlSpec(QUInt(new_width), cvs=Shaped(()) if unknown_new else 0)
    old_source = ControlledViaAnd(positive.subbloq, old_spec)
    source, source_adder = old_source.get_ctrl_system(new_spec)
    old_controls = (
        Register(old_source.ctrl_reg_names[0], unsigned_register_type(old_width)),
    )
    new_controls = (Register("added", unsigned_register_type(new_width)),)
    module, adder = extend_component_control_system(
        record,
        {"h": add_h, "p": add_phase},
        old_controls,
        None if unknown_old else (ControlRegister(old_width, (0,)),),
        new_controls,
        None if unknown_new else (ControlRegister(new_width, (0,)),),
    )
    signature = Signature(
        (*new_controls, *old_controls, *(record.functions[0].registers or ()))
    )
    native = ComponentProgramBuilder(signature, module.functions, name="caller")
    observer: Any = _source(signature)
    old_names = [r.name for r in signature if r.name != "added"]
    native_data = {name: native.inputs[name] for name in old_names}
    source_data = {name: observer.inputs[name] for name in old_names}
    new, rest = adder(native, (native.inputs["added"],), native_data)
    expected_new, expected_rest = source_adder(
        observer.builder, (observer.inputs["added"],), source_data
    )
    parent = native.finalize(
        dict(zip((r.name for r in signature), (*new, *rest), strict=True))
    )
    original = observer.builder.finalize(
        **dict(
            zip(
                (r.name for r in signature),
                (*expected_new, *expected_rest),
                strict=True,
            )
        )
    )
    assert tuple(native_data) == tuple(source_data) == ("q", "spectator")
    assert normalize_trace(_native_trace(parent, "caller")) == normalize_trace(
        _source_trace(observer, original)
    )
    assert normalize_trace(_native_trace(module)) == normalize_trace(
        _source_trace(observer, source.decompose_bloq())
    )
    assert verify_program_record(parent) == ()
    declarations = [f for f in module.functions if not f.blocks]
    assert len(declarations) == (2 if old_width + new_width == 2 else 0)


def test_unknown_single_control_retains_source_refusal() -> None:
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
    from qualtran.symbolics import Shaped

    record, positive = fixture_program(1)
    source = ControlledViaAnd(positive.subbloq, CtrlSpec(QUInt(1), cvs=Shaped(())))
    with pytest.raises(ValueError, match="symbolic"):
        source.decompose_bloq()
    with pytest.raises(ValueError, match="concrete activation"):
        control_component_via_predicate(
            record,
            {"h": add_h, "p": add_phase},
            (Register("ctrl", unsigned_register_type(1)),),
            None,
        )


def test_source_symbolic_binary_tensor_and_physical_protocol_disagree() -> None:
    from qualtran.bloqs.mcmt import And

    np = pytest.importorskip("numpy")

    source = And(*sympy.symbols("cv1 cv2"))
    positive = And(1, 1)
    # The atomic tensor makes every symbolic equality false. The physical
    # protocol instead inserts X only when equality with zero is exactly true.
    assert source.to_clifford_t_circuit() == positive.to_clifford_t_circuit()
    symbolic_tensor = source.tensor_contract()
    assert not np.array_equal(symbolic_tensor, positive.tensor_contract())
    assert np.array_equal(symbolic_tensor[::2], np.eye(4))
    assert np.count_nonzero(symbolic_tensor[1::2]) == 0
