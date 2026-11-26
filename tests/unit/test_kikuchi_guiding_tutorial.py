"""Tutorial word graphs retain the frozen root and one-level expansion."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import sympy

from flagquantum.algorithms.kikuchi_guiding_state import GuidingState
from flagquantum.algorithms.kxor_instance import KXorInstance
from flagquantum.compiler.registers.component_expansion import (
    expand_component_calls_once,
)
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.registers import RegisterSide
from tests.unit.test_component_expansion import _labelled_trace
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_projectile_nu_library import _named_inputs
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def tutorial_instances(symbolic: bool) -> tuple[GuidingState, Any]:
    from qualtran.bloqs.optimization.k_xor_sat.kikuchi_guiding_state import (
        GuidingState as OriginalGuidingState,
    )
    from qualtran.bloqs.optimization.k_xor_sat.kxor_instance import (
        KXorInstance as OriginalInstance,
    )

    if symbolic:
        n, m, k = sympy.symbols("n m k", positive=True, integer=True)
        return (
            GuidingState(KXorInstance.symbolic(n, m, k), 3 * k),
            OriginalGuidingState(OriginalInstance.symbolic(n, m, k), 3 * k),
        )
    arguments: dict[str, Any] = dict(n=20, m=100, k=4, planted_advantage=0.8)
    return (
        GuidingState(
            KXorInstance.random_instance(**arguments, rng=np.random.default_rng(100)),
            12,
        ),
        OriginalGuidingState(
            OriginalInstance.random_instance(
                **arguments, rng=np.random.default_rng(100)
            ),
            12,
        ),
    )


@pytest.mark.parametrize("symbolic", (False, True))
def test_default_tutorial_root_retains_exact_source_word_connections(
    symbolic: bool,
) -> None:
    native, original = tutorial_instances(symbolic)
    graph = native.program()
    assert not verify_program_record(graph)
    assert _named_inputs(_native_trace(graph)) == _named_inputs(
        _source_trace(SourceCalls, original.decompose_bloq())
    )
    assert len(graph.functions[0].blocks[0].operations) == 12


def test_concrete_tutorial_flatten_once_keeps_source_atomic_partitions() -> None:
    native, original = tutorial_instances(False)
    graph = native.program()
    functions = {function.name: function for function in graph.functions}
    flattened, labels = expand_component_calls_once(
        graph,
        predicate=lambda _, operation: all(
            register.side == RegisterSide.THRU
            for register in functions[operation.callee or ""].registers or ()
        ),
    )
    assert not verify_program_record(flattened)
    assert len(flattened.functions[0].blocks[0].operations) == 21
    assert _named_inputs(_labelled_trace(flattened, labels)) == _named_inputs(
        _source_trace(SourceCalls, original.decompose_bloq().flatten_once())
    )


def test_default_concrete_duplicate_dependency_has_the_actual_source_body() -> None:
    from qualtran._infra.data_types import QUInt
    from qualtran.bloqs.arithmetic.lists.has_duplicates import HasDuplicates

    native, _ = tutorial_instances(False)
    graph = native.program()
    candidates = [
        function
        for function in graph.functions
        if tuple(register.name for register in function.registers or ())
        == ("xs", "flag")
        and function.blocks
        and function.registers is not None
        and function.registers[0].shape == (12,)
    ]
    assert len(candidates) == 1
    duplicate = candidates[0]
    dependency = ProgramRecord(
        (
            replace(duplicate, name="main"),
            *(f for f in graph.functions if f.name != "main"),
        )
    )
    compare_layers(dependency, HasDuplicates(12, QUInt(5)), atomic_leaves(dependency))
