"""Hubbard outer controls retain source calls, relative phase and exact cleanup."""

from collections import Counter
from dataclasses import replace
from itertools import product
from math import sqrt
from typing import Any

import pytest

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.hubbard_select import SelectHubbard
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_hubbard_selection_factories import (
    hubbard_selection_factories,
)
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_hubbard_select_component import assert_sparse_equal
from tests.unit.test_hubbard_stored_graph import source_terminals
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_sparse_chemistry_prepare import _branch

np = pytest.importorskip("numpy")
pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CASES = ("zero", "one", "array", "matrix", "separate", "word", "nested")


def _source(case: str, already: bool) -> tuple[Any, Any]:
    from qualtran import CtrlSpec, QBit, QUInt
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        SelectHubbard as Source,
    )

    specs = {
        "zero": CtrlSpec(cvs=0),
        "one": CtrlSpec(cvs=1),
        "array": CtrlSpec(cvs=(1, 0)),
        "matrix": CtrlSpec(cvs=np.array([[0, 1], [1, 0]])),
        "separate": CtrlSpec(qdtypes=(QBit(), QBit()), cvs=(np.array(0), np.array(1))),
        "word": CtrlSpec(qdtypes=QUInt(2), cvs=1),
        "one_array": CtrlSpec(cvs=np.array([0])),
    }
    base = Source(2, 2, control_val=1 if already else None)
    if case == "nested":
        source = base.controlled(specs["zero"]).controlled(specs["array"])
        return source, source.ctrl_spec
    spec = specs[case]
    return base.controlled(spec), spec


def _record(source: Any, depth: int = 0) -> ProgramRecord:
    result = resolve_component_program(
        _library(source, depth), 0, hubbard_selection_factories()
    )
    assert verify_program_record(result) == ()
    return result


def _bind(
    record: ProgramRecord, spare: int = 8
) -> tuple[BoundRegisterProgram, dict[str, tuple[int, ...]], tuple[int, ...], int]:
    cursor = 0
    inputs = []
    ports = {}
    for reg in record.functions[0].registers or ():
        bits: list[int] = []
        for _ in reg.all_idxs():
            width = int(reg.dtype.num_qubits)
            wires = tuple(range(cursor, cursor + width))
            inputs.append(wires)
            bits.extend(wires)
            cursor += width
        ports[reg.name] = tuple(bits)
    clean = tuple(range(cursor, cursor + spare))
    result = bind_register_program(
        record, tuple(inputs), clean=clean, borrowed=(cursor + spare,)
    )
    assert result.outputs == tuple(inputs)
    return result, ports, clean, cursor


@pytest.mark.parametrize(
    "case,already,inverse", tuple(product(CASES, (False, True), (False, True)))
)
def test_actual_control_constructors_exact_recursive_calls(
    case: str, already: bool, inverse: bool
) -> None:
    from qualtran import Adjoint

    source, _ = _source(case, already)
    source = Adjoint(source) if inverse else source
    record = _record(source)
    if case == "nested" and already:
        with pytest.raises(TypeError, match="integer scalar arrays"):
            source.decompose_bloq()
        from qualtran import Adjoint as SourceAdjoint

        forward_source = source.subbloq if inverse else source
        child_source = forward_source.subbloq.controlled()
        child_source = SourceAdjoint(child_source) if inverse else child_source
        functions = {function.name: function for function in record.functions}
        reachable = set()
        pending = [record.functions[0].name]
        while pending:
            name = pending.pop()
            if name in reachable:
                continue
            reachable.add(name)
            pending.extend(
                operation.callee
                for block in functions[name].blocks
                for operation in block.operations
                if operation.callee is not None
            )
        signature = _library(child_source, 0).entries[0].declaration.registers
        candidates = [
            functions[name]
            for name in reachable
            if functions[name].registers == signature
        ]
        assert len(candidates) == 1
        child = candidates[0]
        child_record = ProgramRecord(
            (
                replace(child, name="main"),
                *tuple(
                    function
                    for function in record.functions
                    if function.name not in (child.name, "main")
                ),
            )
        )
        compare_layers(
            child_record,
            child_source,
            source_terminals(child_record)
            | {
                function.name
                for function in child_record.functions
                if function.registers == ()
            },
        )
        return
    compare_layers(
        record,
        source,
        source_terminals(record)
        | {f.name for f in record.functions if f.registers == ()},
    )


@pytest.mark.parametrize("case,already", tuple(product(CASES, (False, True))))
def test_outer_controls_full_complex_branches_and_work_zero(
    case: str, already: bool
) -> None:
    from qualtran import Adjoint

    source, spec = _source(case, already)
    forward, backward = _record(source), _record(Adjoint(source))
    bound, ports, clean, logical = _bind(forward)
    erased, _, _, _ = _bind(backward)
    count = bound.circuit.n_wires
    if type(source).__name__ == "_MultiControlledFromSinglyControlled":
        controls = ports["control"][:1]
    elif type(source).__name__ == "SelectHubbard":
        controls = ports["control"]
    else:
        controls = tuple(w for name in source.ctrl_reg_names for w in ports[name])
    if (
        already
        and type(source).__name__ == "ControlledViaAnd"
        and type(source.subbloq).__name__ == "_MultiControlledFromSinglyControlled"
    ):
        controls = (*controls, *ports["control"][:-1])
    required = tuple(
        bit
        for dtype, values in zip(spec.qdtypes, spec.cvs, strict=True)
        for value in values.flat
        for bit in dtype.to_bits(int(value))
    )
    if (
        already
        and type(source).__name__ == "ControlledViaAnd"
        and type(source.subbloq).__name__ == "_MultiControlledFromSinglyControlled"
    ):
        required = (*required, *source.subbloq.cvs[:-1])
    assert len(controls) == len(required)
    activation = sum(
        int(bit) << (len(required) - 1 - i) for i, bit in enumerate(required)
    )
    operation = SelectHubbard(2, controlled=already)
    base = {name: ports[name] for name, _ in operation.layout}
    if already:
        base["control"] = ports["control"][-1:]
    data = tuple(w for name, _ in operation.layout for w in base[name])
    initial: dict[int, complex] = {}
    active: dict[int, complex] = {}
    inactive: dict[int, complex] = {}
    for pattern, sample in product(range(1 << len(controls)), range(16)):
        value = (sample * 1973) % (1 << len(data))
        if already:
            value = (value & ((1 << (len(data) - 1)) - 1)) | (
                (sample % 2) << (len(data) - 1)
            )
        word = sum(
            ((value >> (len(data) - 1 - i)) & 1) << (count - 1 - wire)
            for i, wire in enumerate(data)
        )
        word |= sum(
            ((pattern >> (len(controls) - 1 - i)) & 1) << (count - 1 - wire)
            for i, wire in enumerate(controls)
        )
        word |= sample % 2  # coherent spectator, outside all clean leases
        amplitude = 1j ** (sample + pattern) / sqrt(16 * (1 << len(controls)))
        initial[word] = amplitude
        (active if pattern == activation else inactive)[word] = amplitude
    work, slots = operation.requirements()
    reference = operation._emit(base, clean[:work], tuple(range(slots)), adjoint=False)
    expected = dict(inactive)
    expected.update(
        _branch(
            reference,
            count,
            (0,) * Counter(g.name for g in reference)["measure"],
            active,
            normalize_measurements=True,
        )
    )
    events = certify_erasure_histories(
        bound.circuit.instructions, n_wires=count, zero_wires=clean
    )
    inverse_events = certify_erasure_histories(
        erased.circuit.instructions, n_wires=count, zero_wires=clean
    )
    positive = SelectHubbard(2, controlled=True)
    positive_work, positive_slots = positive.requirements()
    positive_ports = {**base, "control": base.get("control", controls[:1])}
    child_events = Counter(
        g.name
        for g in positive._emit(
            positive_ports,
            clean[:positive_work],
            tuple(range(positive_slots)),
            adjoint=False,
        )
    )["measure"]
    extra = events - child_events
    assert extra == max(len(controls) - 1, 0) + int(already)
    histories = [(0,) * child_events + bits for bits in product((0, 1), repeat=extra)]
    histories += [(1,) * events, tuple(i % 2 for i in range(events))]
    mask = sum(1 << (count - 1 - wire) for wire in clean)
    for history in histories:
        actual = _branch(
            bound.circuit.instructions,
            count,
            history,
            initial,
            normalize_measurements=True,
        )
        assert_sparse_equal(actual, expected)
        assert sum(abs(a) ** 2 for word, a in actual.items() if word & mask) < 1e-22
    for bits in (0, 1):
        restored = _branch(
            (*bound.circuit.instructions, *erased.circuit.instructions),
            count,
            (bits,) * (events + inverse_events),
            initial,
            normalize_measurements=True,
        )
        assert_sparse_equal(restored, initial)
    live: set[int] = set()
    for event in bound.events:
        (
            live.update(event.wires)
            if event.kind == "allocate"
            else live.difference_update(event.wires)
        )
    assert not live
    if case == "zero" and not already:
        circuit = DynamicCircuit(count)
        circuit.append(FieldOperation(count, bound.circuit.instructions))
        assert circuit.to_ir().instructions == bound.circuit.instructions


@pytest.mark.parametrize("case", ("one_array", "nested"))
def test_single_element_and_nested_controls_retain_exact_source_roles(
    case: str,
) -> None:
    source, _ = _source(case, False)
    record = _record(source)
    compare_layers(
        record,
        source,
        source_terminals(record)
        | {f.name for f in record.functions if f.registers == ()},
    )


def test_original_adjoint_enum_failure_and_zero_constructor_remain_explicit() -> None:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        SelectHubbard as Source,
    )

    source, _ = _source("zero", False)
    with pytest.raises(ValueError, match="SpecializeOnCtrlBit"):
        _library(source.adjoint(), 0)
    with pytest.raises(NotImplementedError, match="control_val=0"):
        Source(2, 2, control_val=0)
    factory = hubbard_selection_factories()[
        "qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard.SelectHubbard"
    ]
    with pytest.raises(NotImplementedError, match="control_val=0"):
        factory("bad", None, {"x_dim": 2, "y_dim": 2, "control_val": 0})


def test_unowned_children_refuse_and_symbolic_controls_stay_declarations() -> None:
    import sympy
    from qualtran import CtrlSpec
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        SelectHubbard as Source,
    )

    from flagquantum.compiler.formats.legacy_component_arguments import (
        ControlSpecification,
    )
    from flagquantum.core.ir.program import FunctionRecord
    from flagquantum.core.ir.program.register_types import quantum_bit_type
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.errors import SerializationError

    factory = hubbard_selection_factories()[
        "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
    ]
    with pytest.raises(SerializationError, match="owned"):
        factory(
            "bad",
            None,
            {
                "subbloq": FunctionRecord(
                    "unknown", (), registers=(Register("q", quantum_bit_type()),)
                ),
                "ctrl_spec": ControlSpecification(((quantum_bit_type(), np.array(0)),)),
            },
        )
    n = sympy.Symbol("n", positive=True, integer=True)
    source = Source(n, n).controlled(CtrlSpec(cvs=0))
    record = _record(source)
    assert not record.functions[0].blocks
    with pytest.raises(ValueError):
        bind_register_program(record, ())


@pytest.mark.parametrize(
    "case,already,inverse",
    (
        ("zero", False, False),
        ("array", True, False),
        ("matrix", False, False),
        ("word", True, True),
    ),
)
@pytest.mark.parametrize("depth", (1, 2))
def test_deeper_source_libraries_preserve_complete_definitions_or_source_codec_refusal(
    case: str, already: bool, inverse: bool, depth: int
) -> None:
    from qualtran import Adjoint

    from flagquantum.compiler.registers.component_definition_match import (
        matches_component_definition,
    )

    source, _ = _source(case, already)
    source = Adjoint(source) if inverse else source
    try:
        library = _library(source, depth)
    except ValueError as error:
        assert "SpecializeOnCtrlBit" in str(error) or (
            depth == 2 and "numpy.uint8" in str(error)
        )
        return
    deep = resolve_component_program(library, 0, hubbard_selection_factories())
    shallow = _record(source)
    assert verify_program_record(deep) == ()
    assert matches_component_definition(deep, deep.functions[0].name, shallow)


@pytest.mark.parametrize(
    "case,already", (("zero", False), ("matrix", False), ("zero", True), ("word", True))
)
def test_external_control_resolution_has_no_source_sdk_dependency(
    case: str, already: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    source, _ = _source(case, already)
    library = _library(source, 0)
    expected = _record(source)
    with monkeypatch.context() as patch:
        for module in ("qualtran", "cirq", "kickmix", "quimb"):
            patch.setitem(sys.modules, module, None)
        record = resolve_component_program(library, 0, hubbard_selection_factories())
        assert record == expected
        bound, _, _, _ = _bind(record)
        assert all(gate.matrix is None for gate in bound.circuit.instructions)
