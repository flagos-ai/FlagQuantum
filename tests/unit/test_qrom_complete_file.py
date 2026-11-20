"""Complete frozen QROM factories and traversal hook dispositions."""

from collections.abc import Callable
from dataclasses import replace
from itertools import product
from typing import Any

import pytest

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.registers.lookup.qrom_resource_model import source_qrom_calls
from flagquantum.compiler.registers.lookup.qrom_table import QromTable
from flagquantum.compiler.registers.lookup.source_qrom import _uniform
from tests.unit.test_qrom_file_source_calls import call_key

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")
pytest.importorskip("qualtran")


def source_factory(name: str) -> Any:
    from qualtran.bloqs.data_loading import qrom

    factory: Callable[[], Any] = getattr(qrom, name)
    return factory()


def native_table(source: Any) -> QromTable:
    """Retain the original factories' explicit selection and target widths."""
    return QromTable(
        shape=source.data_shape,
        selection_bitsizes=source.selection_bitsizes,
        target_bitsizes=source.target_bitsizes,
        target_shapes=source.target_shapes,
        num_controls=source.num_controls,
        datasets=(
            tuple(tuple(map(int, array.flat)) for array in source.data)
            if source.has_data()
            else None
        ),
    )


@pytest.mark.parametrize(
    "name", ("_qrom_small", "_qrom_multi_data", "_qrom_multi_dim", "_qrom_symb")
)
def test_actual_documentation_factories_and_all_cost_categories(name: str) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.resource_counting.generalizers import ignore_split_join
    from sympy import simplify

    source = source_factory(name)
    table = native_table(source)
    assert [(r.name, r.bitsize, r.shape, r.side.name) for r in table.signature] == [
        (r.name, r.bitsize, r.shape, r.side.name) for r in source.signature
    ]
    assert source.adjoint() is source
    assert source.my_static_costs(QECGatesCost()) is NotImplemented
    assert str(source) == (
        f"QROM({table.shape}, {table.target_shapes}, {table.target_bitsizes})"
    )
    for legacy in (False, True):
        model = (
            table.resource_model(legacy_shims=legacy).gates
            if source.has_data()
            else table.resource_formula(legacy_shims=legacy)
        )
        cost = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        for category in (
            "and_bloq",
            "clifford",
            "measurement",
            "rotation",
            "cswap",
            "toffoli",
            "t",
        ):
            assert simplify(getattr(model, category) - getattr(cost, category)) == 0
    peak = (
        table.resource_model().qubits
        if source.has_data()
        else table.resource_formula().logical_qubits
    )
    assert simplify(peak - get_cost_value(source, QubitCount())) == 0
    if source.has_data():
        assert table.datasets is not None
        graph, _ = source.call_graph(max_depth=1, generalizer=ignore_split_join)
        assert source_qrom_calls(
            table.num_controls,
            table.selection_bitsizes,
            table.shape,
            table.target_layout,
            table.datasets,
        ) == {
            call_key(child): attributes["n"]
            for _, child, attributes in graph.edges(data=True)
        }
    else:
        assert source._break_early((), 0, 1) is False
        with pytest.raises(ValueError, match="unavailable"):
            table.bind_packed((), (), ())


@pytest.mark.parametrize("depth", (0, 1, 2))
@pytest.mark.parametrize("name", ("_qrom_small", "_qrom_multi_data", "_qrom_multi_dim"))
def test_actual_concrete_factories_foreign_transport(depth: int, name: str) -> None:
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = source_factory(name)
    proto = bloqs_to_proto(source, max_depth=depth)
    wire = proto.SerializeToString(deterministic=True)
    library = deserialize_component_library(wire)
    assert serialize_component_library(library) == wire
    assert bloqs_from_proto(proto)[0] == source
    assert native_table(bloqs_from_proto(proto)[0]) == native_table(source)


@pytest.mark.parametrize("grouped,controls", tuple(product((False, True), range(5))))
def test_every_row_call_and_pruning_retains_flat_target_comparison(
    grouped: bool, controls: int
) -> None:
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.data_loading.qrom import QROM

    # Equal rows with unequal target cells must not trigger the source pruning.
    data = (
        np.array([[1, 2], [1, 2], [3, 3], [3, 3]])
        if grouped
        else np.array([1, 1, 3, 3])
    )
    targets = ((2,),) if grouped else ((),)
    source = QROM.build_from_data(data, target_shapes=targets, num_controls=controls)
    table = native_table(source)
    for left, right in ((0, 4), (0, 2), (2, 4), (0, 1)):
        assert bool(source._break_early((), left, right)) == _uniform(
            tuple(map(int, data[left:right].flat))
        )
    for row in range(4):
        count = sum(int(value).bit_count() for value in data[row : row + 1].flat)
        assert source.nth_operation_callgraph(selection=row) == {(CNOT(), count)}
    assert table.datasets == (tuple(map(int, data.flat)),)


def test_multiaxis_pruning_uses_original_prefix_and_each_dataset() -> None:
    from qualtran.bloqs.data_loading.qrom import QROM

    first = np.array([[1, 1, 2], [3, 3, 3]])
    second = np.array([[4, 4, 4], [5, 6, 6]])
    source = QROM.build_from_data(first, second)
    for prefix, left, right in (((0,), 0, 2), ((0,), 0, 3), ((1,), 0, 3), ((), 0, 2)):
        assert bool(source._break_early(prefix, left, right)) == all(
            _uniform(tuple(map(int, array[prefix][left:right].flat)))
            for array in (first, second)
        )
    with pytest.raises(IndexError):
        source._break_early((0,), 2, 2)


@pytest.mark.parametrize("name", ("_qrom_small", "_qrom_multi_data", "_qrom_multi_dim"))
def test_actual_factory_primitive_stream_and_typed_ownership(name: str) -> None:
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir import CircuitIR
    from tests.unit.test_qrom_base_foreign_file import table_program
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    source = source_factory(name)
    table = native_table(source)
    program = table_program(table, "documented_lookup")
    inputs = []
    ports = {}
    offset = 0
    for register in table.signature:
        start = offset
        for _ in register.all_idxs():
            width = int(register.bitsize)
            inputs.append(tuple(range(offset, offset + width)))
            offset += width
        ports[register.name] = tuple(range(start, offset))
    work = tuple(range(offset, offset + table.workspace_size))
    bound = bind_register_program(
        program, tuple(inputs), entry="documented_lookup", clean=work
    )
    assert bound.outputs == tuple(inputs)
    # The binder creates fresh classical slots on repeated physical measurements.
    # Compare source key reuse only after checking each feedback uses the most
    # recent measurement of that work wire.
    slots = dict(zip(work, range(len(work)), strict=True))
    renamed = []
    slot_names = {}
    latest = {}
    for instruction in bound.circuit.instructions:
        metadata = dict(instruction.metadata)
        if instruction.name == "measure":
            wire = instruction.wires[0]
            slot = metadata["classical_bit"]
            slot_names[slot] = slots[wire]
            latest[slots[wire]] = slot
            metadata["classical_bit"] = slots[wire]
        if "conditions" in metadata:
            conditions = metadata["conditions"]
            for slot, _ in conditions:
                assert latest[slot_names[slot]] == slot
            metadata["conditions"] = tuple(
                (slot_names[slot], value) for slot, value in conditions
            )
        renamed.append(replace(instruction, metadata=metadata))
    reference = reference_arithmetic_gates(source, ports, work, slots, cirq_style=True)
    assert_source(tuple(renamed), reference)
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
