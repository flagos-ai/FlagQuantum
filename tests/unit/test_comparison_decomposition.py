"""Complete comparison catalog disposition with explicit source graph boundaries."""

from typing import Any, Literal

import pytest

from examples.algorithms import arithmetic_ui_export as native
from examples.algorithms.comparison_decomposition import (
    comparison_catalog_program,
    comparison_score,
)
from flagquantum.compiler.registers.comparison.constant_component import (
    less_than_constant_program,
)
from flagquantum.compiler.registers.comparison.controlled_component import (
    controlled_greater_program,
)
from flagquantum.compiler.registers.comparison.equality_component import (
    equality_constant_program,
    equality_program,
)
from flagquantum.compiler.registers.comparison.half_component import (
    half_comparison_program,
)
from flagquantum.compiler.registers.comparison.ordering_component import (
    greater_than_program,
    less_equal_program,
)
from flagquantum.compiler.registers.comparison.tree_component import (
    mixer_program,
    single_qubit_comparison_program,
)
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from tests.unit.test_arithmetic_ui_export import originals as originals
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_gate_ui_export import _canonical_score

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

EXACT = (
    "greater_than",
    "equals",
    "eq_k",
    "clineardepthgreaterthan_example",
    "lineardepthhalfgreaterthan_small",
    "lineardepthhalfgreaterthanequal_small",
    "lineardepthhalflessthan_small",
    "lineardepthhalflessthanequal_small",
)
# These owners remain authoritative for physical execution. Their immediate
# grouping or independent-node insertion differs from the source SDK import.
GRAPH_BOUNDARIES = {
    "lt_k": (44, 44),
    "leq": (43, 43),
    "bi_qubits_mixer": (8, 18),
    "sq_cmp": (3, 7),
}
REFUSED = ("lt_k_symb", "leq_symb", "gt_k")


def _record(identifier: str) -> ProgramRecord:
    record = comparison_catalog_program(
        native.name(identifier),
        native.signature(identifier),
        native.attributes(identifier),
    )
    assert record is not None and verify_program_record(record) == ()
    return record


@pytest.mark.parametrize("identifier", EXACT)
def test_complete_ordered_calls_operands_types_and_fresh_diagrams(
    originals: tuple[Any, dict[str, Any]], identifier: str
) -> None:
    from qualtran.drawing import get_musical_score_data

    record = _record(identifier)
    composite = originals[1][identifier].decompose_bloq()
    assert _native_trace(record) == _source_trace(
        _source(native.signature(identifier)), composite
    )
    functions = {f.name: f for f in record.functions}
    children = {
        instance.i: instance.bloq for instance, _, _ in composite.iter_bloqsoqs()
    }
    encodings = {"QAny": "opaque", "QBit": "bit", "QUInt": "uint", "QInt": "signed"}
    for index, op in enumerate(record.functions[0].blocks[0].operations):
        actual = functions[op.callee or ""].registers or ()
        source = children[index].signature
        assert [
            (r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in actual
        ] == [(r.name, r.bitsize, r.shape, r.side.value) for r in source]
        for port, original in zip(actual, source, strict=True):
            if identifier == "eq_k" and index == 2 and port.side.name == "RIGHT":
                # Source bb.join(xs) loses the nominal unsigned type. The
                # executable owner intentionally retains the existing uint view.
                assert (
                    port.dtype.encoding == "uint"
                    and type(original.dtype).__name__ == "QAny"
                )
            else:
                assert port.dtype.encoding == encodings[type(original.dtype).__name__]
    fresh = get_musical_score_data(composite)
    expected = {
        "max_x": fresh.max_x,
        "max_y": fresh.max_y,
        **{
            key: [v.json_dict() for v in getattr(fresh, key)]
            for key in ("soqs", "hlines", "vlines")
        },
    }
    actual_score = comparison_score(
        native.name(identifier),
        native.signature(identifier),
        native.attributes(identifier),
    )
    assert actual_score is not None
    assert _canonical_score(dict(actual_score)) == _canonical_score(expected)
    assert native.decomposition(identifier) == record
    assert native.score(identifier) == actual_score


@pytest.mark.parametrize("identifier", GRAPH_BOUNDARIES)
def test_native_owner_grouping_difference_is_explicit(
    originals: tuple[Any, dict[str, Any]], identifier: str
) -> None:
    record = _record(identifier)
    composite = originals[1][identifier].decompose_bloq()
    actual = _native_trace(record)
    expected = _source_trace(_source(native.signature(identifier)), composite)
    assert (len(actual), len(expected)) == GRAPH_BOUNDARIES[identifier]
    assert actual != expected
    assert native.decomposition(identifier) == record


def test_every_concrete_graph_reuses_existing_executable_owner() -> None:
    expected = {
        "lt_k": less_than_constant_program(8, 5),
        "greater_than": greater_than_program(4),
        "equals": equality_program(native.signature("equals")[0].dtype),
        "eq_k": equality_constant_program(4, 13),
        "leq": less_equal_program(4, 8),
        "bi_qubits_mixer": mixer_program(),
        "sq_cmp": single_qubit_comparison_program(),
        "clineardepthgreaterthan_example": controlled_greater_program(
            native.signature("clineardepthgreaterthan_example")[1].dtype
        ),
    }
    relations: dict[str, Literal["gt", "ge", "lt", "le"]] = {
        "greaterthan": "gt",
        "greaterthanequal": "ge",
        "lessthan": "lt",
        "lessthanequal": "le",
    }
    for suffix, relation in relations.items():
        identifier = f"lineardepthhalf{suffix}_small"
        expected[identifier] = half_comparison_program(
            native.signature(identifier)[0].dtype, relation
        )
    assert expected.keys() == set(EXACT) | GRAPH_BOUNDARIES.keys()
    for identifier, owner in expected.items():
        assert _record(identifier) == owner


@pytest.mark.parametrize("identifier", REFUSED)
def test_original_decomposition_refusals_remain_atomic(
    originals: tuple[Any, dict[str, Any]], identifier: str
) -> None:
    from qualtran import DecomposeNotImplementedError, DecomposeTypeError

    error = DecomposeNotImplementedError if identifier == "gt_k" else DecomposeTypeError
    with pytest.raises(error):
        originals[1][identifier].decompose_bloq()
    assert (
        comparison_catalog_program(
            native.name(identifier),
            native.signature(identifier),
            native.attributes(identifier),
        )
        is None
    )
    assert native.decomposition(identifier) is None
