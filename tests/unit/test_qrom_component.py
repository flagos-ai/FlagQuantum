"""Source-exact typed lookup calls and dirty-target measured execution."""

from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.lookup.qrom_component import qrom_program
from flagquantum.compiler.registers.lookup.qrom_table import QromTable
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_table(table: QromTable) -> Any:
    from qualtran.bloqs.data_loading import QROM

    data = tuple(
        np.asarray(values).reshape((*table.shape, *shape))
        for values, shape in zip(table.datasets or (), table.target_shapes, strict=True)
    )
    return QROM(
        data,
        table.selection_bitsizes,
        table.target_bitsizes,
        table.target_shapes,
        table.num_controls,
    )


@pytest.mark.parametrize("controls", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "data", [(0, 1, 3, 2), (2, 2, 3, 3), (1, 1, 1, 1), (0, 0, 0, 0)]
)
def test_qrom_exact_root_children_and_complete_dirty_target_operator(
    controls: int, data: tuple[int, ...]
) -> None:
    table = QromTable((4,), (2,), (2,), ((),), controls, (data,))
    source = source_table(table)
    record = qrom_program(table)
    assert verify_program_record(record) == ()
    compare_layers(record, source, atomic_leaves(record))
    inputs = (
        *((tuple(range(controls)),) if controls else ()),
        tuple(range(controls, controls + 2)),
        tuple(range(controls + 2, controls + 4)),
    )
    bound = bind_register_program(
        record,
        inputs,
        clean=tuple(range(controls + 4, controls + 4 + table.workspace_size)),
    )
    from tests.unit.test_multi_control_component import MeasurementSlots
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates

    ports = {
        reg.name: wires for reg, wires in zip(table.signature, inputs, strict=True)
    }
    expected_gates = reference_arithmetic_gates(
        source,
        ports,
        tuple(range(controls + 4, controls + 4 + table.workspace_size)),
        MeasurementSlots(bound.circuit.instructions),
        keep_pauli_leaves=True,
    )
    from tests.unit.test_register_subtraction import assert_source

    assert_source(bound.circuit.instructions, expected_gates)
    if controls <= 1:
        expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
        torch.testing.assert_close(
            logical_operator(bound.circuit, controls + 4), expected, atol=1e-11, rtol=0
        )


@pytest.mark.parametrize(
    "shape,target_shape", [((2, 2), ()), ((2, 3), ()), ((4,), (2,)), ((2, 2), (2,))]
)
def test_multiaxis_and_array_words_retain_full_source_call_graph(
    shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> None:
    count = int(np.prod((*shape, *target_shape)))
    data = tuple((i * 3 + i // 2) % 4 for i in range(count))
    table = QromTable(
        shape,
        tuple((n - 1).bit_length() for n in shape),
        (2,),
        (target_shape,),
        datasets=(data,),
    )
    source = source_table(table)
    record = qrom_program(table)
    compare_layers(record, source, atomic_leaves(record))
    assert verify_program_record(record) == ()
