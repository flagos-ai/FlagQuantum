"""The complete multiplication file retains recursive source graphs and exceptions."""

from collections import Counter
from dataclasses import replace
from typing import Any

import pytest

from examples.algorithms import field_ui_export as native
from examples.algorithms.component_ui_dependencies import ComponentFunction
from examples.algorithms.field_product_decomposition import (
    field_product_child_decomposition,
)
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_basic_gate_child_decomposition import _label, _source_score
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_field_ui_export import originals as originals
from tests.unit.test_register_component_calls import SourceCalls
from tests.unit.test_register_gate_ui_export import _canonical_score

CONCRETE = tuple(native.INSTANCES)[:1] + tuple(native.INSTANCES)[2:9]
pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_walk(
    identifier: str, source: Any, *, catalog: Any = native
) -> list[dict[str, Any]]:
    """Observe every named source child, including the frozen inverse defect."""
    from qualtran import DecomposeNotImplementedError, DecomposeTypeError

    record = catalog.decomposition(identifier)
    assert record is not None and verify_program_record(record) == ()
    functions = {f.name: f for f in record.functions}
    entry = record.functions[0].name
    pending = [(entry, source)]
    visited: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    while pending:
        name, source = pending.pop(0)
        if name in visited:
            assert visited[name] == source
            continue
        visited[name] = source
        function = functions[name]
        row: dict[str, Any] = {"entry": name, "source_type": type(source).__name__}
        assert [
            (r.name, r.bitsize, r.shape, r.side.value) for r in function.registers or ()
        ] == [(r.name, r.bitsize, r.shape, r.side.value) for r in source.signature]
        try:
            composite = source.decompose_bloq()
        except (DecomposeNotImplementedError, DecomposeTypeError) as error:
            row.update(disposition="source_atomic_refusal", error=type(error).__name__)
            rows.append(row)
            continue
        actual = _native_trace(record, name)
        expected = _source_trace(SourceCalls, composite)
        observed, original = [], []
        for left, right in zip(actual, expected, strict=True):
            assert isinstance(left, tuple) and isinstance(right, tuple)
            observed.append((left[0], dict(left[1]), left[2]))
            original.append((right[0], dict(right[1]), right[2]))
        exact = observed == original
        if not exact:
            assert identifier in {"gf2mulviakaratsuba", "gf16_inverse"}
            assert type(source).__name__ == "SynthesizeLRCircuit" and source.is_adjoint
            assert (
                _source_trace(SourceCalls, source.adjoint().decompose_bloq())
                == expected
            )
        row.update(
            disposition="exact_graph" if exact else "original_ignored_adjoint_defect",
            native_calls=len(actual),
            source_calls=len(expected),
        )
        if name == entry:
            drawing = catalog.score(identifier)
        else:
            node = ComponentFunction(
                identifier, name, record, identifier, catalog._child_presentation
            )
            assert node.presentation is not None, (identifier, name, source)
            label = _label(source)
            if not exact:
                label += "†"
            assert node.presentation[0] == label
            row["qualified"] = label
            drawing = node.score()
        assert isinstance(drawing, dict)
        expected_score = _source_score(composite)
        if exact:
            assert _canonical_score(drawing) == _canonical_score(expected_score), (
                identifier,
                source,
                name,
            )
            row["drawing"] = "source_exact"
        else:
            assert _canonical_score(drawing) != _canonical_score(expected_score)
            row["drawing"] = "native_exact_inverse_original_defect_retained"
        rows.append(row)
        children = {b.i: b.bloq for b, _, _ in composite.iter_bloqsoqs()}
        pending.extend(
            (op.callee or "", children[i])
            for i, op in enumerate(function.blocks[0].operations)
        )
    return rows


@pytest.mark.parametrize("identifier", CONCRETE)
def test_complete_source_file_recursive_calls_classes_and_fresh_drawings(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    rows = source_walk(identifier, originals[1][identifier])
    counts = Counter(row["disposition"] for row in rows)
    assert counts["exact_graph"] > 0
    assert counts["original_ignored_adjoint_defect"] == (
        1 if identifier == "gf2mulviakaratsuba" else 0
    )


def test_original_symbolic_product_refusal(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    from qualtran import DecomposeTypeError

    with pytest.raises(DecomposeTypeError):
        originals[1]["gf2_multiplication_symbolic"].decompose_bloq()
    assert native.decomposition("gf2_multiplication_symbolic") is None


def test_source_child_labels_reject_changed_primitive_body() -> None:
    identifier = "gf2mulviakaratsuba"
    record = native.decomposition(identifier)
    assert record is not None
    changed = replace(
        record,
        functions=tuple(
            replace(
                f,
                blocks=tuple(
                    replace(
                        b,
                        operations=tuple(
                            (
                                replace(op, opcode="cz")
                                if op.opcode == "cx"
                                else (
                                    replace(op, operands=op.operands[::-1])
                                    if op.opcode == "ccx"
                                    else op
                                )
                            )
                            for op in b.operations
                        ),
                    )
                    for b in f.blocks
                ),
            )
            for f in record.functions
        ),
    )
    assert changed != record
    rejected = 0
    for function in record.functions:
        arguments = (
            function.name,
            native.signature(identifier),
            native.attributes(identifier),
        )
        if field_product_child_decomposition(record, *arguments) is not None:
            assert field_product_child_decomposition(changed, *arguments) is None
            rejected += 1
    assert rejected > 0
