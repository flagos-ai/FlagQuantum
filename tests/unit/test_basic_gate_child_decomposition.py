"""Complete recursive Basic Gates calls retain source wiring and child glyphs."""

import json
from dataclasses import replace
from typing import Any

import pytest

from examples.algorithms.basic_gate_child_decomposition import child_decomposition
from examples.algorithms.basic_gate_decomposition import decomposition_program
from examples.algorithms.component_ui_dependencies import ComponentFunction
from flagquantum.compiler.registers.component_expansion import _has_call_decomposition
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_gate_ui_export import (
    DECOMPOSITION_EXAMPLES,
    EXAMPLES,
    _canonical_score,
)
from tests.unit.test_register_gate_ui_export import (
    originals as originals,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _source_score(composite: Any) -> dict[str, Any]:
    from qualtran.drawing import get_musical_score_data

    source = get_musical_score_data(composite)
    payload = {
        "max_x": source.max_x,
        "max_y": source.max_y,
        **{
            key: [item.json_dict() for item in getattr(source, key)]
            for key in ("soqs", "hlines", "vlines")
        },
    }
    decoded: dict[str, Any] = json.loads(
        json.dumps(payload, default=lambda value: value.item())
    )
    return decoded


def _label(bloq: Any) -> str:
    return (
        _label(bloq.subbloq) + "†"
        if type(bloq).__name__ == "Adjoint"
        else type(bloq).__name__
    )


@pytest.mark.parametrize(
    "identifier",
    (*DECOMPOSITION_EXAMPLES, "swap_large", "cswap_large", "approx_cswap_large"),
)
def test_complete_recursive_calls_source_identities_and_child_drawings(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    root = EXAMPLES[identifier]
    program = decomposition_program(root)
    assert program is not None
    functions = {function.name: function for function in program.functions}
    entry = program.functions[0].name
    pending = [(entry, originals[1][identifier])]
    visited: dict[str, Any] = {}
    while pending:
        name, source = pending.pop(0)
        if name in visited:
            assert visited[name] == source
            continue
        visited[name] = source
        function = functions[name]
        assert [
            (port.name, port.bitsize, port.shape, port.side.value)
            for port in function.registers or ()
        ] == [
            (port.name, port.bitsize, port.shape, port.side.value)
            for port in source.signature
        ]
        try:
            composite = source.decompose_bloq()
        except Exception as error:
            assert type(error).__name__ in {
                "DecomposeTypeError",
                "DecomposeNotImplementedError",
            }
            continue
        assert _has_call_decomposition(function, functions)
        selected = replace(
            program,
            functions=(function, *(f for f in program.functions if f.name != name)),
        )
        assert _native_trace(selected, name) == _source_trace(
            _source(Signature(function.registers or ())), composite
        )
        if name != entry:
            node = ComponentFunction(identifier, name, program, root)
            assert node.presentation is not None
            assert node.presentation[0] == _label(source)
            try:
                expected = _source_score(composite)
            except IndexError:
                assert identifier == "approx_cswap_large"
            else:
                actual = node.score()
                assert isinstance(actual, dict)
                assert _canonical_score(actual) == _canonical_score(expected)
        children = {
            instance.i: instance.bloq for instance, _, _ in composite.iter_bloqsoqs()
        }
        assert len(children) == len(function.blocks[0].operations)
        pending.extend(
            (operation.callee or "", children[index])
            for index, operation in enumerate(function.blocks[0].operations)
        )


def test_changed_phase_cannot_reuse_a_known_child_drawing() -> None:
    root = EXAMPLES["swz_small"]
    record = decomposition_program(root)
    assert record is not None
    candidate = next(
        function
        for function in record.functions
        if tuple(port.name for port in function.registers or ()) == ("ctrl", "x", "y")
    )
    assert child_decomposition(record, candidate.name, root) is not None
    changed = []
    for function in record.functions:
        blocks = tuple(
            replace(
                block,
                operations=tuple(
                    (
                        replace(operation, opcode="tdg")
                        if operation.opcode == "t"
                        else operation
                    )
                    for operation in block.operations
                ),
            )
            for block in function.blocks
        )
        changed.append(replace(function, blocks=blocks))
    mutated = replace(record, functions=tuple(changed))
    assert child_decomposition(mutated, candidate.name, root) is None
    node = ComponentFunction("changed", candidate.name, mutated, root)
    assert "operation" not in node.attributes()
