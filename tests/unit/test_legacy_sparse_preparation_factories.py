"""Resolve complete sparse PREPARE constructors and explicit native inverses."""

import sys
from unittest.mock import patch

import pytest

from flagquantum import Circuit
from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_sparse_preparation_factories import (
    native_sparse_preparation_factories,
)
from flagquantum.core.ir.program import verify_program_record
from tests.unit.preparation_reference import preparation_terminals
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_qubitization_walk_component import (
    reference_runtime as reference_runtime,
)
from tests.unit.test_sparse_chemistry_prepare_component import (
    bind_used_workspace,
    native_pair,
)
from tests.unit.test_sparse_chemistry_prepare_component import (
    frozen_sparse_preparation_source as frozen_sparse_preparation_source,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("log", (0, 1))
@pytest.mark.parametrize("inverse", (False, True))
def test_complete_constructor_depths_calls_and_sdk_free_append(
    log: int, inverse: bool
) -> None:
    from qualtran.bloqs.chemistry.sparse.prepare import PrepareSparse

    rows = ((0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0), (0, 0, 1, 1))
    original = PrepareSparse(
        4,
        4,
        2,
        tuple(tuple(reversed(row)) for row in rows),
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        rows,
        (0, 1, 1, 0),
        (1, 0, 0, 1),
        (0, 3, 1, 0),
        log_block_size=log,
    )
    source = original.adjoint() if inverse else original
    direct, _ = bind_used_workspace(
        native_pair(original)[int(inverse)], allow_workspace_outputs=True
    )
    for depth in (0, 1):
        record = resolve_component_program(
            _library(source, depth), 0, native_sparse_preparation_factories()
        )
        assert not verify_program_record(record)
        leaves = preparation_terminals(
            record,
            native_boundaries=(
                f.name
                for f in record.functions
                if f.name.endswith("native_qroam_erase_leaf")
            ),
        )
        compare_layers(record, source, leaves)
        with patch.dict(sys.modules, {"qualtran": None, "cirq": None}):
            bound, _ = bind_used_workspace(record, allow_workspace_outputs=True)
            assert bound.circuit.instructions == direct.circuit.instructions
            circuit = Circuit(bound.circuit.n_wires).append(
                _BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires)
            )
            assert circuit.to_ir().instructions == bound.circuit.instructions


@pytest.mark.parametrize("inverse", (False, True))
def test_original_walk_65_row_twelve_bit_constructor(inverse: bool) -> None:
    from qualtran.bloqs.qubitization.qubitization_walk_operator import (
        _walk_op_chem_sparse,
    )

    original = _walk_op_chem_sparse.make().reflect.prepare_gate
    assert original.num_non_zero == 65
    assert original.num_bits_state_prep == 12
    source = original.adjoint() if inverse else original
    for depth in (0, 1):
        record = resolve_component_program(
            _library(source, depth), 0, native_sparse_preparation_factories()
        )
        assert not verify_program_record(record)
        leaves = preparation_terminals(
            record,
            native_boundaries=(
                f.name
                for f in record.functions
                if f.name.endswith("native_qroam_erase_leaf")
            ),
        )
        compare_layers(record, source, leaves)
