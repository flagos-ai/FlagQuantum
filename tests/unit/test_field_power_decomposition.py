"""Complete square and inverse files retain power/product graphs and source defects."""

from collections import Counter
from dataclasses import replace
from typing import Any

import pytest

from examples.algorithms import field_ui_export as native
from tests.unit.test_field_product_decomposition import source_walk
from tests.unit.test_field_ui_export import originals as originals

CONCRETE = ("gf16_square", "gf16_inverse")
pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("identifier", CONCRETE)
def test_complete_power_source_recursive_calls_and_drawings(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    rows = source_walk(identifier, originals[1][identifier])
    counts = Counter(row["disposition"] for row in rows)
    assert counts["exact_graph"] > 0
    assert counts["original_ignored_adjoint_defect"] == 0


def test_symbolic_square_retains_original_refusal(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    from qualtran._infra.bloq import DecomposeTypeError

    with pytest.raises(DecomposeTypeError):
        originals[1]["gf2_square_symbolic"].decompose_bloq()
    assert native.decomposition("gf2_square_symbolic") is None


def test_power_child_labels_reject_changed_primitive_operands() -> None:
    from examples.algorithms.field_power_decomposition import field_power_child

    identifier = "gf16_inverse"
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
                                replace(op, operands=op.operands[::-1])
                                if op.opcode in {"cx", "ccx"}
                                else op
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
    rejected = 0
    for f in record.functions:
        arguments = (
            f.name,
            native.signature(identifier),
            native.attributes(identifier),
        )
        if field_power_child(record, *arguments) is not None:
            assert field_power_child(changed, *arguments) is None
            rejected += 1
    assert rejected > 0


def test_literal_inverse_roles_survive_opaque_function_renaming() -> None:
    from examples.algorithms.field_power_decomposition import field_power_child

    record = native.decomposition("gf16_inverse")
    assert record is not None
    names = {f.name: f"opaque_{i}" for i, f in enumerate(record.functions)}
    changed = replace(
        record,
        functions=tuple(
            replace(
                f,
                name=names[f.name],
                blocks=tuple(
                    replace(
                        b,
                        operations=tuple(
                            replace(op, callee=names[op.callee]) if op.callee else op
                            for op in b.operations
                        ),
                    )
                    for b in f.blocks
                ),
            )
            for f in record.functions
        ),
    )
    for function in record.functions:
        before = field_power_child(
            record, function.name, native.signature("gf16_inverse"), {}
        )
        after = field_power_child(
            changed, names[function.name], native.signature("gf16_inverse"), {}
        )
        assert (None if before is None else before[0]) == (
            None if after is None else after[0]
        )
