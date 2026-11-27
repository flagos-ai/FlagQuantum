"""Original QROM construction is distinct from validating physical table storage."""

from dataclasses import replace
from importlib import import_module
from importlib.metadata import version
from typing import Any

import pytest
import sympy

from flagquantum.compiler.formats.legacy_qrom_factories import qrom_factories
from flagquantum.compiler.registers.lookup.qrom_component import qrom_program
from flagquantum.compiler.registers.lookup.qrom_declaration import QromDeclaration
from flagquantum.compiler.registers.lookup.qrom_table import QromTable
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.signatures import Signature
from tests._reference_dependencies import unavailable_reference

try:
    import kickmix
    import numpy as np
    from numpy.typing import ArrayLike

    source_module = import_module("qualtran.bloqs.data_loading.qrom")
    source_resources = import_module("qualtran.resource_counting")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def reference_runtime() -> None:
    assert version("qualtran") == "0.7.0"
    assert getattr(kickmix, "__version__", None) == "0.1" and hasattr(
        kickmix, "Simulator"
    )


def assert_ports(native: Signature, source: Any) -> None:
    assert [(r.name, r.bitsize, r.shape, r.side.value) for r in native] == [
        (r.name, r.bitsize, r.shape, r.side.value) for r in source
    ]
    for owned, foreign in zip(native, source, strict=True):
        if type(foreign.dtype).__name__ == "BQUInt":
            assert owned.dtype.encoding == "uint"
            assert (
                dict(owned.dtype.parameters)["bound"] == foreign.dtype.iteration_length
            )
        else:
            assert type(foreign.dtype).__name__ == "QAny"
            assert owned.dtype.encoding == "opaque"


@pytest.mark.parametrize(
    "data,targets",
    (
        (([0, 1, 3],), ((),)),
        (([[0, 1], [1, 0]],), ((2,),)),
        (([[0, 1], [1, 0]], [1, 0]), ((2,), ())),
        ((1,), ((),)),
        (([0.9, 2.8, 1.1],), ((),)),
        (([-1, -2],), ((),)),
    ),
)
@pytest.mark.parametrize("controls", (0, 1, 3))
def test_complete_literal_declarations_materialize_existing_layout(
    data: tuple[ArrayLike, ...], targets: tuple[tuple[int, ...], ...], controls: int
) -> None:
    source = source_module.QROM.build_from_data(
        *data, target_shapes=targets, num_controls=controls
    )
    native = QromDeclaration.from_data(
        *data, target_shapes=targets, num_controls=controls
    )
    assert native.data_shape == source.data_shape
    assert native.dataset_shapes == tuple(row.shape for row in source.data)
    assert native.datasets == tuple(tuple(map(int, row.flat)) for row in source.data)
    assert native.selection_bitsizes == source.selection_bitsizes
    assert native.target_bitsizes == source.target_bitsizes
    assert native.has_data() == source.has_data()
    assert native.is_symbolic() == source.is_symbolic()
    assert_ports(native.signature, source.signature)
    assert native.materialize() == QromTable.from_data(
        *data, target_shapes=targets, num_controls=controls
    )
    assert native.static_qubit_count() == source_resources.get_cost_value(
        source, source_resources.QubitCount()
    )


@pytest.mark.parametrize(
    "case",
    ("inconsistent_rows", "wrong_target_axis", "scalar_sentinel", "negative_control"),
)
def test_source_deferred_intake_and_strict_materialization(case: str) -> None:
    data: tuple[ArrayLike, ...]
    targets: tuple[tuple[int, ...], ...] = ()
    controls = 0
    if case == "inconsistent_rows":
        data = ([1, 2], [1, 2, 3])
    elif case == "wrong_target_axis":
        data, targets = (np.arange(6).reshape(3, 2),), ((3,),)
    elif case == "scalar_sentinel":
        data = (1, [1, 2])
    else:
        data, controls = ([1, 0],), -1
    source = source_module.QROM.build_from_data(
        *data, target_shapes=targets, num_controls=controls
    )
    native = QromDeclaration.from_data(
        *data, target_shapes=targets, num_controls=controls
    )
    assert native.datasets == tuple(tuple(map(int, row.flat)) for row in source.data)
    assert native.selection_bitsizes == source.selection_bitsizes
    assert native.target_shapes == source.target_shapes
    if case == "inconsistent_rows":
        for read in (lambda: source.data_shape, lambda: native.data_shape):
            with pytest.raises(ValueError, match="same shape"):
                read()
    else:
        assert native.data_shape == source.data_shape
        assert_ports(native.signature, source.signature)
        assert native.static_qubit_count() == source_resources.get_cost_value(
            source, source_resources.QubitCount()
        )
    with pytest.raises(ValueError):
        native.materialize()


@pytest.mark.parametrize("symbolic", (False, True))
@pytest.mark.parametrize("target", ((), (2,)))
def test_unknown_shapes_replacement_and_signatures(
    symbolic: bool, target: tuple[int, ...]
) -> None:
    n = sympy.Symbol("n", positive=True, integer=True) if symbolic else 3
    source = source_module.QROM.build_from_bitsize(n, (2,), target_shapes=(target,))
    native = QromDeclaration.from_shape(n, (2,), target_shapes=(target,))
    assert native.data_shape == source.data_shape
    assert native.has_data() == source.has_data() is False
    assert native.is_symbolic() == source.is_symbolic()
    assert_ports(native.signature, source.signature)
    assert native.static_qubit_count() == source_resources.get_cost_value(
        source, source_resources.QubitCount()
    )
    if not symbolic:
        values = np.arange(3 * (2 if target else 1)).reshape((3,) + target) % 4
        loaded, original = native.with_data(values), source.with_data(values)
        assert loaded.datasets == tuple(
            tuple(map(int, row.flat)) for row in original.data
        )
        assert loaded.materialize() == QromTable.from_data(
            values, target_bitsizes=(2,), target_shapes=(target,)
        )
        for replace_data in (native.with_data, source.with_data):
            with pytest.raises(AssertionError):
                replace_data(np.zeros((4,) + target, dtype=int))
            with pytest.raises(AssertionError):
                replace_data(np.full((3,) + target, 4))
        assert not native.has_data()


def test_literals_copy_input_and_preserve_explicit_selection_metadata() -> None:
    values = np.array([0, 1])
    native = QromDeclaration.from_data(values)
    values[0] = 9
    assert native.datasets == ((0, 1),)
    source = source_module.QROM(
        data_or_shape=(np.array([0, 1]),),
        selection_bitsizes=(2,),
        target_bitsizes=(1,),
    )
    widened = replace(native, selection_bitsizes=(2,))
    assert_ports(widened.signature, source.signature)
    assert widened.materialize().selection_bitsizes == (2,)


def test_factory_retains_source_wrong_suffix_declaration_without_a_body() -> None:
    native = QromDeclaration.from_data(
        np.arange(6).reshape(3, 2), target_shapes=((3,),)
    )
    factory = next(iter(qrom_factories().values()))
    function = factory(
        "main",
        tuple(native.signature),
        {
            "data_or_shape": (np.arange(6).reshape(3, 2),),
            "selection_bitsizes": np.array([2]),
            "target_bitsizes": np.array([3]),
            "target_shapes": ((3,),),
        },
    )
    assert isinstance(function, FunctionRecord)
    assert function.registers == tuple(native.signature) and not function.blocks


def test_valid_factory_still_uses_the_existing_executable_component() -> None:
    declaration = QromDeclaration.from_data([0, 1, 3, 2])
    table = declaration.materialize()
    factory = next(iter(qrom_factories().values()))
    actual = factory(
        "main",
        tuple(declaration.signature),
        {
            "data_or_shape": ([0, 1, 3, 2],),
            "selection_bitsizes": (2,),
            "target_bitsizes": (2,),
        },
    )
    from flagquantum.compiler.formats.legacy_word_factories import _rename_program

    assert actual == _rename_program(qrom_program(table), "main")


def test_raw_uint64_factory_retains_high_bit_calls_and_complex_sparse_state() -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum import Circuit
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
    from tests.unit.sparse_register_execution import execute_sparse_register

    values = np.array([1 << 63, 0], dtype=np.uint64)
    source = source_module.QROM((values,), (1,), (64,))
    captured = import_register_emitter(source)
    factory = next(iter(qrom_factories().values()))
    record = factory(
        "main",
        None,
        {
            "data_or_shape": (values,),
            "selection_bitsizes": (1,),
            "target_bitsizes": (64,),
        },
    )
    from flagquantum.core.ir.program import ProgramRecord

    assert isinstance(record, ProgramRecord)
    bound = bind_register_program(record, ((0,), tuple(range(1, 65))))
    assert bound.circuit.instructions == captured.snapshots[0].instructions
    assert [gate.name for gate in bound.circuit.instructions] == ["x", "cx", "x"]
    assert captured.requirements() == (0, 0)
    ir = Circuit(66).append(FieldOperation(66, bound.circuit.instructions)).to_ir()
    # Caller-owned spectator is the least significant bit and remains correlated.
    before = {0: 0.5 + 0j, 1: 0.5j, (1 << 65) | 3: -0.5, (1 << 63) | 1: -0.5j}
    expected = {
        word ^ ((1 << 64) if word >> 65 == 0 else 0): value
        for word, value in before.items()
    }
    assert execute_sparse_register(ir, before, seed=0) == expected


@pytest.mark.parametrize(
    "kind", ("two", "fractional", "matrix", "text", "nonreflexive")
)
@pytest.mark.parametrize("controls", (0, 1))
def test_raw_pruning_precedes_integer_loads_in_calls_channels_and_models(
    kind: str, controls: int
) -> None:
    from itertools import product

    import torch
    from qualtran.resource_counting.generalizers import ignore_split_join

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum import Circuit
    from flagquantum.compiler.formats.legacy_component_library import (
        deserialize_component_library,
        serialize_component_library,
    )
    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.compiler.registers.lookup.qrom_declaration_resources import (
        declaration_qrom_calls,
        declaration_qrom_gate_cost,
    )
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_component_hierarchy_control import compare_layers
    from tests.unit.test_legacy_arithmetic_factories import _library
    from tests.unit.test_multi_control_component import MeasurementSlots, atomic_leaves
    from tests.unit.test_multiplexed_source_binding import logical_operator
    from tests.unit.test_qrom_file_source_calls import call_key
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    if kind == "two":
        data = np.array([0.1, 0.2])
    elif kind == "text":
        data = np.array(["1", "01", "2", "02"])
    elif kind == "nonreflexive":
        data = np.array([complex(1, float("nan")), complex(1, float("nan"))])
    else:
        data = np.array([0.1, 0.2, 1.1, 1.2])
        if kind == "matrix":
            data = data.reshape(2, 2)
    widths = tuple((n - 1).bit_length() for n in data.shape)
    source = source_module.QROM((data,), widths, (2,), num_controls=controls)
    declaration = QromDeclaration.from_arrays(
        data,
        selection_bitsizes=widths,
        target_bitsizes=(2,),
        num_controls=controls,
    )
    record = qrom_program(declaration)
    compare_layers(record, source, atomic_leaves(record))
    for depth in (0, 1, 2):
        try:
            library = _library(source, depth)
        except KeyError:
            # The frozen non-reflexive attrs hash can lose its own root ID.
            assert kind == "nonreflexive"
            continue
        wire = serialize_component_library(library)
        assert serialize_component_library(deserialize_component_library(wire)) == wire
        actual = resolve_component_program(library, 0, qrom_factories())
        assert actual.functions[0].registers == tuple(declaration.signature)
        # Renaming the root is the existing library adapter's only difference.
        from flagquantum.compiler.formats.legacy_word_factories import _rename_program

        assert actual == _rename_program(record, actual.functions[0].name)
    logical = controls + sum(widths) + 2
    inputs: list[tuple[int, ...]] = [(0,)] if controls else []
    start = controls
    for size in (*widths, 2):
        inputs.append(tuple(range(start, start + size)))
        start += size
    work = tuple(range(logical, logical + max(0, sum(widths) + controls - 1)))
    bound = bind_register_program(record, tuple(inputs), clean=work)
    ports = {
        reg.name: wires
        for reg, wires in zip(declaration.signature, inputs, strict=True)
    }
    source_gates = reference_arithmetic_gates(
        source,
        ports,
        work,
        MeasurementSlots(bound.circuit.instructions),
        keep_pauli_leaves=True,
    )
    assert_source(bound.circuit.instructions, source_gates)
    ir = (
        Circuit(logical + len(work))
        .append(FieldOperation(logical + len(work), bound.circuit.instructions))
        .to_ir()
    )
    operator = torch.zeros((1 << logical, 1 << logical), dtype=torch.complex128)
    words = tuple(map(int, data.flat))
    for column in range(1 << logical):
        active = not controls or bool(column >> (logical - 1))
        address = (column >> 2) & (len(words) - 1)
        result = column ^ (words[address] if active else 0)
        operator[result, column] = 1
    torch.testing.assert_close(
        logical_operator(ir, logical), operator, atol=8e-12, rtol=0
    )
    extended = (
        Circuit(ir.n_wires + 1)
        .append(FieldOperation(ir.n_wires + 1, ir.instructions))
        .to_ir()
    )
    vector = torch.tensor(
        [complex(i + 1, 3 - i) for i in range(1 << (logical + 1))],
        dtype=torch.complex128,
    )
    vector /= vector.norm()
    initial = torch.zeros(1 << extended.n_wires, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for index, amplitude in enumerate(vector):
        word, spectator = index >> 1, index & 1
        active = not controls or bool(word >> (logical - 1))
        address = (word >> 2) & (len(words) - 1)
        result = word ^ (words[address] if active else 0)
        initial[(word << (len(work) + 1)) | spectator] = amplitude
        expected[(result << (len(work) + 1)) | spectator] = amplitude
    events = sum(g.name == "measure" for g in ir.instructions)
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, extended.instructions, history),
            expected / 2 ** (events / 2),
            atol=8e-12,
            rtol=0,
        )
    if kind == "nonreflexive":
        from collections import Counter

        source_calls = dict(
            source.build_call_graph(source_resources.SympySymbolAllocator())
        )
        declared: Counter[str] = Counter()
        for child, count in source_calls.items():
            for category, value in (
                source_resources.get_cost_value(child, source_resources.QECGatesCost())
                .asdict()
                .items()
            ):
                declared[category] += count * value
        assert declaration_qrom_gate_cost(declaration).asdict() == {
            key: value for key, value in declared.items() if value
        }
        assert declaration.static_qubit_count() == source.my_static_costs(
            source_resources.QubitCount()
        )
        expected_calls = {
            call_key(child): count for child, count in source_calls.items()
        }
    else:
        assert (
            declaration_qrom_gate_cost(declaration).asdict()
            == source_resources.get_cost_value(
                source, source_resources.QECGatesCost()
            ).asdict()
        )
        graph, _ = source.call_graph(max_depth=1, generalizer=ignore_split_join)
        expected_calls = {
            call_key(child): edge["n"] for _, child, edge in graph.edges(data=True)
        }
    assert declaration_qrom_calls(declaration) == expected_calls


@pytest.mark.parametrize("value", (float("nan"), float("inf")))
def test_failed_source_integer_load_keeps_its_logical_header(value: float) -> None:
    data = np.array([value, 0.0])
    source = source_module.QROM((data,), (1,), (1,))
    qubit = source_module.cirq.LineQubit(0)
    with pytest.raises((ValueError, OverflowError)):
        next(source._load_nth_data((0,), target0_=np.array([qubit])))
    factory = next(iter(qrom_factories().values()))
    function = factory(
        "main",
        None,
        {
            "data_or_shape": (data,),
            "selection_bitsizes": (1,),
            "target_bitsizes": (1,),
        },
    )
    assert isinstance(function, FunctionRecord) and not function.blocks
    assert function.registers is not None
    assert_ports(Signature(function.registers), source.signature)


def test_actual_wrong_suffix_library_roundtrip_and_physical_refusal() -> None:
    from flagquantum.compiler.formats.legacy_component_library import (
        deserialize_component_library,
        serialize_component_library,
    )
    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from tests.unit.test_legacy_arithmetic_factories import _library

    source = source_module.QROM.build_from_data(
        np.arange(6).reshape(3, 2), target_shapes=((3,),)
    )
    library = _library(source, 0)
    wire = serialize_component_library(library)
    assert serialize_component_library(deserialize_component_library(wire)) == wire
    record = resolve_component_program(library, 0, qrom_factories())
    assert not record.functions[0].blocks
    with pytest.raises(ValueError, match="requires component bodies"):
        bind_register_program(
            record,
            ((0, 1), (2, 3, 4), (5, 6, 7), (8, 9, 10)),
            entry=record.functions[0].name,
        )
    with pytest.raises(IndexError):
        tuple(
            source._load_nth_data(
                (0,),
                target0_=np.asarray(
                    source_module.cirq.LineQubit.range(9), dtype=object
                ).reshape(3, 3),
            )
        )


@pytest.mark.parametrize("field", ("selection", "control"))
def test_boolean_source_index_metadata_uses_the_existing_bit_storage(
    field: str,
) -> None:
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir.program import ProgramRecord
    from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
    from tests.unit.test_register_subtraction import assert_source

    widths = (True,) if field == "selection" else (1,)
    controls = True if field == "control" else 0
    data = np.array([0, 1])
    source = source_module.QROM((data,), widths, (1,), num_controls=controls)
    declaration = QromDeclaration.from_arrays(
        data, selection_bitsizes=widths, target_bitsizes=(1,), num_controls=controls
    )
    assert_ports(declaration.signature, source.signature)
    table = declaration.materialize()
    captured = import_register_emitter(source)
    record = next(iter(qrom_factories().values()))(
        "main",
        None,
        {
            "data_or_shape": (data,),
            "selection_bitsizes": widths,
            "target_bitsizes": (1,),
            "num_controls": controls,
        },
    )
    assert isinstance(record, ProgramRecord)
    logical = 2 + int(bool(controls))
    bound = bind_register_program(
        record,
        tuple((i,) for i in range(logical)),
        clean=tuple(range(logical, logical + table.workspace_size)),
    )
    assert_source(bound.circuit.instructions, captured.snapshots[0].instructions)


def test_original_flat_array_identity_collision_does_not_merge_native_headers() -> None:
    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from tests.unit.test_legacy_arithmetic_factories import _library

    matrix = source_module.QROM((np.arange(4).reshape(2, 2),), (2,), (2,))
    flat = source_module.QROM((np.arange(4),), (2,), (2,))
    assert matrix == flat  # Original attrs identity ignores ndarray shape.
    assert matrix.signature != flat.signature
    first = resolve_component_program(_library(matrix, 0), 0, qrom_factories())
    second = resolve_component_program(_library(flat, 0), 0, qrom_factories())
    assert first.functions[0].registers != second.functions[0].registers
    assert not first.functions[0].blocks and second.functions[0].blocks


@pytest.mark.parametrize("controls", (2, 3))
@pytest.mark.parametrize("pattern", (0, 1))
def test_constant_table_retains_source_multi_control_workspace_and_erasure(
    controls: int, pattern: int
) -> None:
    import torch

    from flagquantum.core.ir import CircuitIR
    from tests.unit.test_multi_control_component import MeasurementSlots
    from tests.unit.test_multiplexed_source_binding import logical_operator
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    data = np.full(4, pattern, dtype=int)
    source = source_module.QROM((data,), (2,), (1,), num_controls=controls)
    table = QromTable.from_data(data, num_controls=controls)
    assert table.workspace_size == controls - 1
    logical = controls + 3
    work = tuple(range(logical, logical + table.workspace_size))
    gates = table.bind_packed(tuple(range(logical)), work, tuple(range(len(work))))
    expected_gates = reference_arithmetic_gates(
        source,
        {
            "control": tuple(range(controls)),
            "selection": (controls, controls + 1),
            "target0_": (controls + 2,),
        },
        work,
        MeasurementSlots(gates),
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    assert_source(gates, expected_gates)
    expected = torch.zeros((1 << logical, 1 << logical), dtype=torch.complex128)
    for column in range(1 << logical):
        enabled = column >> 3 == (1 << controls) - 1
        expected[column ^ (pattern if enabled else 0), column] = 1
    torch.testing.assert_close(
        logical_operator(CircuitIR(logical + len(work), gates), logical),
        expected,
        atol=8e-12,
        rtol=0,
    )
    assert sum(g.name == "measure" for g in gates) == controls - 1
    assert sum(g.name in {"t", "tdg"} for g in gates) == 4 * (controls - 1)


@pytest.mark.parametrize("controls", (0, 1, 2))
@pytest.mark.parametrize("case", ("ignored_nan", "trailing_singleton", "empty_target"))
def test_source_lazy_loads_only_visited_declared_target_cells(
    controls: int, case: str
) -> None:
    from itertools import product

    import torch

    from flagquantum import Circuit
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir.program import ProgramRecord, verify_program_record
    from flagquantum.core.ir.program.register_verifier import executable_register_width
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_component_hierarchy_control import compare_layers
    from tests.unit.test_multi_control_component import MeasurementSlots, atomic_leaves
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    selection: tuple[int, ...]
    targets: tuple[tuple[int, ...], ...]
    if case == "ignored_nan":
        data = np.array([[0.0, np.nan], [1.0, np.nan], [3.0, np.nan]])
        selection, targets, width = (2,), ((1,),), 2
    elif case == "trailing_singleton":
        data = np.array([[1], [2]])
        selection, targets, width = (1, 0), ((),), 2
    else:
        # The raw columns still affect pruning despite no emitted load words.
        data = np.array([[0, 1], [2, 3]])
        selection, targets, width = (1,), ((0,),), 2
    source = source_module.QROM((data,), selection, (width,), targets, controls)
    declaration = QromDeclaration.from_arrays(
        data,
        selection_bitsizes=selection,
        target_bitsizes=(width,),
        target_shapes=targets,
        num_controls=controls,
    )
    if case == "trailing_singleton":
        # NumPy 2.5.4 removed the source's former size-one array int coercion.
        with pytest.raises(TypeError, match="0-dimensional"):
            reference_arithmetic_gates(
                source,
                {
                    **({"control": tuple(range(controls))} if controls else {}),
                    "selection": (controls,),
                    "target0_": (controls + 1, controls + 2),
                },
                tuple(range(controls + 3, controls + 5)),
                {},
                cirq_style=True,
            )
        with pytest.raises(TypeError, match="0-dimensional"):
            qrom_program(declaration)
        return
    record = qrom_program(declaration)
    assert verify_program_record(record) == ()
    factory_record = next(iter(qrom_factories().values()))(
        "main",
        None,
        {
            "data_or_shape": (data,),
            "selection_bitsizes": selection,
            "target_bitsizes": (width,),
            "target_shapes": targets,
            "num_controls": controls,
        },
    )
    from flagquantum.compiler.formats.legacy_word_factories import _rename_program

    assert isinstance(factory_record, ProgramRecord)
    assert factory_record == _rename_program(record, "main")
    if case == "empty_target":
        with pytest.raises(ValueError, match="iteration dimensions are 0"):
            source.decompose_bloq()
    else:
        compare_layers(record, source, atomic_leaves(record))
    inputs: list[tuple[int, ...]] = []
    logical = 0
    ports = {}
    for reg in declaration.signature:
        wires = tuple(
            range(
                logical,
                logical
                + executable_register_width(reg.dtype) * int(np.prod(reg.shape)),
            )
        )
        logical += len(wires)
        ports[reg.name] = wires
        # Entry inputs contain one tuple per register-array element.
        word_width = int(reg.bitsize)
        inputs.extend(
            tuple(wires[i : i + word_width]) for i in range(0, len(wires), word_width)
        )
    work_count = sum(selection) + controls - 1
    work = tuple(range(logical, logical + max(work_count, 0)))
    bound = bind_register_program(record, tuple(inputs), clean=work)
    gates = bound.circuit.instructions
    expected_gates = reference_arithmetic_gates(
        source,
        ports,
        work,
        MeasurementSlots(gates),
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    assert_source(gates, expected_gates)
    # Compare every small raw measurement history on a complex entangled input,
    # including dirty target words and an untouched spectator wire.
    from examples.algorithms.binary_field_arithmetic import FieldOperation

    ir = (
        Circuit(bound.circuit.n_wires + 1)
        .append(FieldOperation(bound.circuit.n_wires + 1, gates))
        .to_ir()
    )
    vector = torch.tensor(
        [complex(i + 1, 2 - i) for i in range(1 << (logical + 1))],
        dtype=torch.complex128,
    )
    selector_width = sum(selection)
    target_width = 0 if case == "empty_target" else width
    # Bounded selectors promise only valid addresses. Gate equality above also
    # retains the source's unspecified action on unused binary encodings.
    for index in range(len(vector)):
        address = ((index >> 1) >> target_width) & ((1 << selector_width) - 1)
        if address >= len(data):
            vector[index] = 0
    vector /= vector.norm()
    initial = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for index, amplitude in enumerate(vector):
        word, spectator = index >> 1, index & 1
        address = (word >> target_width) & ((1 << selector_width) - 1)
        enabled = (
            not controls
            or word >> (selector_width + target_width) == (1 << controls) - 1
        )
        loaded = 0
        if enabled and case != "empty_target" and address < len(data):
            loaded = int(data[address, 0])
        result = word ^ loaded
        initial[(word << (len(work) + 1)) | spectator] = amplitude
        expected[(result << (len(work) + 1)) | spectator] = amplitude
    events = sum(g.name == "measure" for g in gates)
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, ir.instructions, history),
            expected / 2 ** (events / 2),
            atol=8e-12,
            rtol=0,
        )
    assert sum(g.name in {"t", "tdg"} for g in gates) == 4 * events


@pytest.mark.parametrize("controls", (-1, 0, 1, 2, 3))
@pytest.mark.parametrize(
    "case", ("zero_word", "ignored_nan", "negative_load", "empty_target")
)
def test_declaration_resource_queries_do_not_require_physical_integer_tables(
    controls: int, case: str
) -> None:
    from flagquantum.compiler.registers.lookup.qrom_declaration_resources import (
        declaration_qrom_calls,
        declaration_qrom_gate_cost,
    )
    from tests.unit.test_qrom_file_source_calls import call_key

    targets = (
        ((0,),)
        if case == "empty_target"
        else ((1,),) if case == "ignored_nan" else ((),)
    )
    width = 0 if case == "zero_word" else 2
    data = (
        np.array([[0.0, np.nan], [1.0, np.nan], [3.0, np.nan]])
        if case == "ignored_nan"
        else (
            np.array([-1, 2, 0, 1])
            if case == "negative_load"
            else (
                np.array([[0, 1], [2, 3]])
                if case == "empty_target"
                else np.zeros(2, dtype=int)
            )
        )
    )
    selection = ((len(data) - 1).bit_length(),)
    source = source_module.QROM((data,), selection, (width,), targets, controls)
    native = QromDeclaration.from_arrays(
        data,
        selection_bitsizes=selection,
        target_bitsizes=(width,),
        target_shapes=targets,
        num_controls=controls,
    )
    if case == "zero_word" and controls < 0:
        for quote in (
            lambda: source.build_call_graph(source_resources.SympySymbolAllocator()),
            lambda: declaration_qrom_calls(native),
        ):
            with pytest.raises(ValueError):
                quote()
        return
    declared = dict(source.build_call_graph(source_resources.SympySymbolAllocator()))
    source_generalizers = import_module("qualtran.resource_counting.generalizers")
    expected = {
        call_key(child): count
        for child, count in declared.items()
        if source_generalizers.ignore_split_join(child) is not None
    }
    assert declaration_qrom_calls(native) == expected
    # Aggregate exactly the same direct child protocol. A root SDK cache may
    # fail independently for non-reflexive arrays; no primitive count is used.
    original = source_resources.GateCounts()
    for child, count in declared.items():
        original += (
            source_resources.get_cost_value(child, source_resources.QECGatesCost())
            * count
        )
    assert declaration_qrom_gate_cost(native).asdict() == original.asdict()


@pytest.mark.parametrize("field", ("selection", "control"))
@pytest.mark.parametrize("value", (sympy.Integer(1), sympy.Integer(2)))
def test_exact_symbolic_integer_index_metadata_remains_executable(
    field: str, value: Any
) -> None:
    from flagquantum.compiler.formats.legacy_word_factories import _rename_program
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir.program import ProgramRecord
    from tests.unit.test_multi_control_component import MeasurementSlots
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    selection = (value,) if field == "selection" else (1,)
    controls = value if field == "control" else 0
    data = np.array([0, 1])
    source = source_module.QROM((data,), selection, (2,), num_controls=controls)
    declaration = QromDeclaration.from_arrays(
        data, selection_bitsizes=selection, target_bitsizes=(2,), num_controls=controls
    )
    record = next(iter(qrom_factories().values()))(
        "main",
        None,
        {
            "data_or_shape": (data,),
            "selection_bitsizes": selection,
            "target_bitsizes": (2,),
            "num_controls": controls,
        },
    )
    assert isinstance(record, ProgramRecord)
    assert record == _rename_program(qrom_program(declaration), "main")
    logical = int(sum(selection) + controls + 2)
    work = tuple(range(logical, logical + max(0, int(sum(selection) + controls - 1))))
    ports = {}
    offset = 0
    for reg in source.signature:
        ports[reg.name] = tuple(range(offset, offset + int(reg.total_bits())))
        offset += int(reg.total_bits())
    bound = bind_register_program(record, tuple(ports.values()), clean=work)
    expected = reference_arithmetic_gates(
        source,
        ports,
        work,
        MeasurementSlots(bound.circuit.instructions),
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    assert_source(bound.circuit.instructions, expected)


@pytest.mark.parametrize("width", (sympy.Integer(1), sympy.Integer(2)))
def test_symbolic_integer_target_retains_source_declared_model(width: Any) -> None:
    from flagquantum.compiler.registers.lookup.qrom_declaration_resources import (
        declaration_qrom_calls,
        declaration_qrom_gate_cost,
    )
    from tests.unit.test_qrom_file_source_calls import call_key

    data = np.array([0, 1])
    source = source_module.QROM((data,), (1,), (width,))
    declaration = QromDeclaration.from_arrays(
        data, selection_bitsizes=(1,), target_bitsizes=(width,)
    )
    calls = dict(source.build_call_graph(source_resources.SympySymbolAllocator()))
    assert declaration_qrom_calls(declaration) == {
        call_key(child): n for child, n in calls.items()
    }
    expected = source_resources.get_cost_value(source, source_resources.QECGatesCost())
    assert declaration_qrom_gate_cost(declaration).asdict() == expected.asdict()
    assert declaration_qrom_gate_cost(declaration).clifford == 3
    constant = QromDeclaration.from_arrays(
        np.zeros(2, dtype=int), selection_bitsizes=(1,), target_bitsizes=(width,)
    )
    constant_source = source_module.QROM((np.zeros(2, dtype=int),), (1,), (width,))
    with pytest.raises(import_module("qualtran").DecomposeTypeError):
        constant_source.build_call_graph(source_resources.SympySymbolAllocator())
    with pytest.raises(ValueError, match="parameterized"):
        declaration_qrom_calls(constant)


def test_empty_raw_constructor_retains_scalar_logical_calls_without_public_carrier() -> (
    None
):
    from flagquantum.compiler.registers.lookup.qrom_declaration_resources import (
        declaration_qrom_calls,
        declaration_qrom_gate_cost,
    )
    from flagquantum.core.ir.program import ProgramRecord, verify_program_record

    source = source_module.QROM((), (), ())
    native = QromDeclaration.from_arrays(selection_bitsizes=(), target_bitsizes=())
    assert_ports(native.signature, source.signature)
    assert native.has_data() == source.has_data() is True
    assert (
        native.static_qubit_count()
        == source.my_static_costs(source_resources.QubitCount())
        == -1
    )
    assert (
        declaration_qrom_calls(native)
        == dict(source.build_call_graph(source_resources.SympySymbolAllocator()))
        == {}
    )
    assert (
        declaration_qrom_gate_cost(native).asdict()
        == source_resources.get_cost_value(
            source, source_resources.QECGatesCost()
        ).asdict()
        == {}
    )
    record = next(iter(qrom_factories().values()))(
        "main",
        None,
        {"data_or_shape": (), "selection_bitsizes": (), "target_bitsizes": ()},
    )
    assert isinstance(record, ProgramRecord) and verify_program_record(record) == ()
    # This is an empty logical component; no zero-wire public Circuit is added.
    assert record.functions[0].registers == ()
