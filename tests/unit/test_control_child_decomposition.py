"""Shared control descendants retain activation, measured erasure and source glyphs."""

from dataclasses import replace
from typing import Any

import pytest

from examples.algorithms.control_child_decomposition import control_child_decomposition
from flagquantum.compiler.registers.control.multi_control_component import (
    multi_control_x_program,
)
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score_data import register_score_data
from tests.unit.test_basic_gate_child_decomposition import _label, _source_score
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_gate_ui_export import _canonical_score

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    ("count", "value"),
    [(count, value) for count in (2, 3, 4) for value in range(1 << count)],
)
def test_all_small_control_polarities_recursive_source_calls_and_drawings(
    count: int, value: int
) -> None:
    from qualtran import DecomposeNotImplementedError, DecomposeTypeError
    from qualtran.bloqs.mcmt.multi_control_pauli import MultiControlX

    values = tuple((value >> i) & 1 for i in reversed(range(count)))
    record = multi_control_x_program(values)
    functions = {function.name: function for function in record.functions}
    pending = [(record.functions[0].name, MultiControlX(values))]
    visited: dict[str, Any] = {}
    while pending:
        name, source = pending.pop(0)
        if name in visited:
            assert visited[name] == source
            continue
        visited[name] = source
        function = functions[name]
        try:
            composite = source.decompose_bloq()
        except (DecomposeTypeError, DecomposeNotImplementedError):
            continue
        qualified = control_child_decomposition(record, name, {"val": value})
        assert qualified is not None, (count, value, name, source)
        label, template, symbols = qualified
        assert label == _label(source)
        assert _native_trace(record, name) == _source_trace(
            _source(Signature(function.registers or ())), composite
        )
        actual = register_score_data(
            component_register_score(template, template.functions[0].name),
            register_wire_symbol=symbols,
        )
        assert _canonical_score(dict(actual)) == _canonical_score(
            _source_score(composite)
        )
        children = {
            instance.i: instance.bloq for instance, _, _ in composite.iter_bloqsoqs()
        }
        pending.extend(
            (operation.callee or "", children[index])
            for index, operation in enumerate(function.blocks[0].operations)
        )


def test_measured_feedback_mutation_rejects_every_control_wrapper() -> None:
    record = multi_control_x_program((1, 0, 1, 0))
    changed = replace(
        record,
        functions=tuple(
            replace(
                function,
                blocks=tuple(
                    replace(
                        block,
                        operations=tuple(
                            (
                                replace(
                                    operation,
                                    condition_clauses=tuple(
                                        tuple(
                                            replace(c, equals=not c.equals)
                                            for c in clause
                                        )
                                        for clause in operation.condition_clauses
                                    ),
                                )
                                if operation.condition_clauses
                                else operation
                            )
                            for operation in block.operations
                        ),
                    )
                    for block in function.blocks
                ),
            )
            for function in record.functions
        ),
    )
    assert changed != record
    rejected = set()
    for function in record.functions:
        selected = control_child_decomposition(record, function.name, {"val": 10})
        if selected is not None and selected[0] in {
            "MultiControlX",
            "ControlledViaAnd",
            "CtrlSpecAnd†",
            "MultiAnd†",
        }:
            assert (
                control_child_decomposition(changed, function.name, {"val": 10}) is None
            )
            rejected.add(selected[0])
    assert rejected == {
        "MultiControlX",
        "ControlledViaAnd",
        "CtrlSpecAnd†",
        "MultiAnd†",
    }
