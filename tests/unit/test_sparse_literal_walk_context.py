"""Retain the original sparse walk ABI without inventing an eraser body."""

import sys
from importlib import import_module
from typing import Any
from unittest.mock import patch

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import_module("numpy")
    import_module("cirq")
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


@pytest.mark.parametrize("values", (1, 0, (1, 0)))
@pytest.mark.parametrize("direction", ("forward", "before", "after"))
def test_actual_sparse_walk_constructor_control_orders(
    literal_context: tuple[Any, Any], values: int | tuple[int, ...], direction: str
) -> None:
    from qualtran import Adjoint, CtrlSpec

    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.compiler.formats.legacy_qubitization_walk_factories import (
        qubitization_walk_factories,
    )
    from flagquantum.compiler.formats.legacy_word_factories import _rename_program
    from flagquantum.compiler.registers.component_definition_match import (
        _named_definitions,
    )
    from flagquantum.core.ir.program import ProgramRecord
    from flagquantum.ecosystem._default_linear_combination_oracles import _signature
    from flagquantum.ecosystem.cirq._sparse_chemistry_library import (
        sparse_chemistry_library_factories,
    )
    from tests.unit.test_legacy_arithmetic_factories import _library

    source, context = literal_context
    # Supply the actual captured literal PREPARE, including its missing original
    # eraser; the measured mathematical inverse factory is a separate profile.
    definitions = context.reflection_pair[0].functions
    ports = tuple(_signature(source.prepare.signature))
    roots = [f for f in definitions if f.registers == ports and f.blocks]
    assert len(roots) == 1
    prep = ProgramRecord(_named_definitions(context.reflection_pair[0], roots[0].name))
    factories = sparse_chemistry_library_factories()
    factories.update(
        {
            "qualtran.bloqs.chemistry.sparse.prepare.PrepareSparse": lambda name, registers, arguments: _rename_program(
                prep, name
            ),
            "qualtran.bloqs.block_encoding.lcu_block_encoding.SelectBlockEncoding": lambda name, registers, arguments: _rename_program(
                context.block_pair[0], name
            ),
        }
    )
    factories.update(
        qubitization_walk_factories(
            lambda _: context,
            adjoint_factory=factories["qualtran._infra.adjoint.Adjoint"],
        )
    )
    selected = Adjoint(source) if direction == "before" else source
    selected = selected.controlled(CtrlSpec(cvs=values))
    if direction == "after":
        selected = Adjoint(selected)
    record = resolve_component_program(_library(selected, 0), 0, factories)
    assert_calls(record, selected)
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
        for record in context.pair(controlled=True, source_control_recipes=True):
            assert verify_program_record(record) == ()


@pytest.mark.parametrize("inverse", (False, True))
def test_actual_sparse_source_positive_control_recipe(
    literal_context: tuple[Any, Any], inverse: bool
) -> None:
    from qualtran import Adjoint, CtrlSpec

    from tests.unit.test_component_hierarchy_control import compare_layers
    from tests.unit.test_multi_control_component import atomic_leaves

    source, context = literal_context
    blocks = context.positive_source_block_pair
    assert blocks is not None
    positive_block = source.block_encoding.get_ctrl_system(CtrlSpec())[0]
    expected_block = positive_block.adjoint() if inverse else positive_block
    compare_layers(
        blocks[int(inverse)], expected_block, atomic_leaves(blocks[int(inverse)])
    )
    expected = source.controlled()
    if inverse:
        expected = Adjoint(expected)
    record = context.pair(controlled=True, source_control_recipes=True)[int(inverse)]
    assert_calls(record, expected)
    compare_layers(record, expected, atomic_leaves(record))
    with pytest.raises(ValueError, match="requires component bodies"):
        bind_used_workspace(record, allow_workspace_outputs=True)
