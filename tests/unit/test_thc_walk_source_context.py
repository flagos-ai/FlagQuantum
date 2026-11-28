"""Compare actual THC factory graphs without inventing missing physical leaves."""

import sys
from hashlib import sha1
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.ecosystem._thc_walk_component_library import capture_thc_walk_context
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_factorization_source_closure import assert_returns
from tests.unit.test_projectile_nu_library import _named_inputs
from tests.unit.test_register_component_calls import SourceCalls
from tests.unit.test_sparse_chemistry_prepare_component import bind_used_workspace
from tests.unit.test_thc_selection_source_constructors import (
    source_module as source_module,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_walk_sources(source_module: Any) -> None:
    from qualtran.bloqs.chemistry.thc import walk_operator
    from qualtran.bloqs.qubitization import qubitization_walk_operator

    for module, digest in (
        (walk_operator, "30de4297b0027dee7b552ccc31682fd3e5e27e12"),
        (qubitization_walk_operator, "117515efb1e42c58b0fab451c454fbd756585f74"),
    ):
        raw = Path(module.__file__).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


@pytest.fixture(scope="module")
def blocked(source_module: Any) -> tuple[Any, Any]:
    from attrs import evolve
    from qualtran.bloqs.chemistry.thc.prepare_test import build_random_test_integrals
    from qualtran.bloqs.chemistry.thc.walk_operator import get_walk_operator_for_thc_ham

    source = get_walk_operator_for_thc_ham(
        *build_random_test_integrals(2, 2, seed=7), 8, 8, kr1=2, kr2=2
    )
    source = evolve(
        source,
        block_encoding=evolve(
            source.block_encoding, prepare=evolve(source.prepare, log_block_size=1)
        ),
    )
    return source, capture_thc_walk_context(source)


def assert_calls(record: ProgramRecord, original: Any) -> None:
    from qualtran import CompositeBloq

    graph = (
        original if isinstance(original, CompositeBloq) else original.decompose_bloq()
    )
    assert verify_program_record(record) == ()
    assert _named_inputs(_native_trace(record)) == _named_inputs(
        _source_trace(SourceCalls, graph)
    )
    assert_returns(record, graph)


@pytest.mark.parametrize("positive", (False, True))
def test_complete_factory_children_and_directional_calls(
    blocked: tuple[Any, Any], positive: bool
) -> None:
    from qualtran import CtrlSpec

    source, context = blocked
    block, reflection = source.block_encoding, source.reflect
    if positive:
        block = block.get_ctrl_system(CtrlSpec())[0]
        reflection = reflection.get_ctrl_system(CtrlSpec())[0]
    blocks = context.positive_block_pair if positive else context.block_pair
    reflections = (
        context.positive_reflection_pair if positive else context.reflection_pair
    )
    assert blocks is not None and reflections is not None
    assert_calls(blocks[0], block)
    if positive:
        # Controlled(SelectBlockEncoding).adjoint() changes specialization:
        # its child uses ctrl, unlike forward SelectTHC(control_val=1)'s
        # control bystander. Compare the actual stored forward graph inverse;
        # never claim equality with that distinct source constructor recipe.
        assert_calls(blocks[1], block.decompose_bloq().adjoint())
    else:
        assert_calls(blocks[1], block.adjoint())
    assert_calls(reflections[0], reflection)
    assert reflections[0] == reflections[1]
    for record in context.pair(controlled=positive):
        assert verify_program_record(record) == ()
        with pytest.raises(ValueError, match="requires component bodies"):
            bind_used_workspace(record, allow_workspace_outputs=True)
    if not positive:
        assert_calls(context.pair()[0], source)
        assert_calls(context.pair()[1], source.adjoint())


def test_zero_bank_retains_original_missing_body(source_module: Any) -> None:
    from qualtran.bloqs.chemistry.thc.prepare_test import build_random_test_integrals
    from qualtran.bloqs.chemistry.thc.walk_operator import get_walk_operator_for_thc_ham

    source = get_walk_operator_for_thc_ham(
        *build_random_test_integrals(2, 2, seed=7), 8, 8, kr1=2, kr2=2
    )
    assert source.prepare.log_block_size == 0
    with pytest.raises(KeyError, match="junk_target0_"):
        source.prepare.decompose_bloq()
    context = capture_thc_walk_context(source)
    assert not context.reflection_pair[0].functions[0].blocks
    assert_calls(context.block_pair[0], source.block_encoding)
    assert_calls(context.pair()[0], source)
    with pytest.raises(ValueError, match="requires component bodies"):
        bind_used_workspace(context.pair()[0], allow_workspace_outputs=True)


def test_captured_dependency_graph_needs_no_source_sdk(
    blocked: tuple[Any, Any],
) -> None:
    _, context = blocked
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        for positive in (False, True):
            for record in context.pair(controlled=positive):
                assert verify_program_record(record) == ()
