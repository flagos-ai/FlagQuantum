"""Complete polynomial addition factories retain typed graphs and atomic views."""

from collections import Counter
from dataclasses import replace
from typing import Any

import pytest

from examples.algorithms import field_ui_export as native
from examples.algorithms.field_polynomial_decomposition import field_polynomial_child
from tests.unit.test_field_product_decomposition import source_walk
from tests.unit.test_field_ui_export import originals as originals

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CONCRETE = ("gf2_poly_4_8_add_k", "gf2_poly_4_8_add")


@pytest.mark.parametrize("identifier", CONCRETE)
def test_complete_polynomial_source_recursive_calls_and_fresh_drawings(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    rows = source_walk(identifier, originals[1][identifier])
    counts = Counter(row["disposition"] for row in rows)
    assert counts["exact_graph"] > 0
    assert counts["original_ignored_adjoint_defect"] == 0


@pytest.mark.parametrize(
    "identifier",
    (
        "gf_poly_split",
        "gf_poly_join",
        "gf2_poly_add_k_symbolic",
        "gf2_poly_add_symbolic",
    ),
)
def test_complete_original_symbolic_and_atomic_refusals(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    from qualtran import DecomposeTypeError

    with pytest.raises(DecomposeTypeError):
        originals[1][identifier].decompose_bloq()
    assert native.decomposition(identifier) is None


@pytest.mark.parametrize("identifier", CONCRETE)
def test_coefficient_labels_reject_changed_primitive_bodies(identifier: str) -> None:
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
                                replace(op, opcode="z")
                                if op.opcode == "x"
                                else (
                                    replace(op, operands=op.operands[::-1])
                                    if op.opcode == "cx"
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
    rejected = 0
    for f in record.functions:
        arguments = (
            f.name,
            native.signature(identifier),
            native.attributes(identifier),
        )
        if field_polynomial_child(record, *arguments) is not None:
            assert field_polynomial_child(changed, *arguments) is None
            rejected += 1
    assert rejected > 0
