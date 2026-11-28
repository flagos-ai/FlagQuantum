"""Retain the original sparse walk ABI without inventing an eraser body."""

import sys
from importlib import import_module
from typing import Any
from unittest.mock import patch

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import_module("numpy")
    import_module("qualtran")
    import_module("kickmix")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

from flagquantum.core.ir.program import verify_program_record
from flagquantum.ecosystem._sparse_walk_component_library import (
    capture_sparse_walk_context,
)
from tests.unit.test_sparse_chemistry_prepare_component import bind_used_workspace
from tests.unit.test_thc_selection_source_constructors import (
    source_module as source_module,
)
from tests.unit.test_thc_walk_source_context import assert_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def literal_context(source_module: Any) -> tuple[Any, Any]:
    from qualtran.bloqs.qubitization.qubitization_walk_operator import (
        _walk_op_chem_sparse,
    )

    source = _walk_op_chem_sparse.make()
    assert source.prepare.num_non_zero == 65
    return source, capture_sparse_walk_context(source, literal_source=True)


@pytest.mark.parametrize("inverse", (False, True))
def test_actual_sparse_factory_literal_calls_and_narrow_abi(
    literal_context: tuple[Any, Any], inverse: bool
) -> None:
    source, context = literal_context
    assert context.signature == context.source_signature
    assert_calls(
        context.block_pair[int(inverse)],
        source.block_encoding.adjoint() if inverse else source.block_encoding,
    )
    assert_calls(context.reflection_pair[0], source.reflect)
    record = context.pair()[int(inverse)]
    assert_calls(record, source.adjoint() if inverse else source)
    with pytest.raises(ValueError, match="requires component bodies"):
        bind_used_workspace(record, allow_workspace_outputs=True)


def test_literal_context_retains_no_source_sdk_dependency(
    literal_context: tuple[Any, Any],
) -> None:
    _, context = literal_context
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        for record in context.pair():
            assert verify_program_record(record) == ()
