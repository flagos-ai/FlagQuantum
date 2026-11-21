"""Complete binary addition source graphs use the existing executable owner."""

from typing import Any

import pytest

from examples.algorithms import field_ui_export as native
from examples.algorithms.field_addition_decomposition import (
    field_addition_catalog_program,
    field_addition_score,
)
from flagquantum.compiler.registers.finite_field.addition_component import (
    field_addition_program,
)
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.finite_field_addition import binary_field_encoding
from tests.unit.test_basic_gate_child_decomposition import _source_score
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_field_ui_export import originals as originals
from tests.unit.test_register_component_calls import SourceCalls
from tests.unit.test_register_gate_ui_export import _canonical_score

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("identifier", ("gf16_addition", "gf16_add_k"))
def test_complete_factory_graph_fresh_drawing_and_executable_owner(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    source = originals[1][identifier].decompose_bloq()
    record = native.decomposition(identifier)
    assert record is not None and verify_program_record(record) == ()
    assert record == field_addition_program(
        native.signature(identifier)[0].dtype,
        constant=1 if identifier == "gf16_add_k" else None,
    )
    assert _native_trace(record) == _source_trace(SourceCalls, source)
    actual = native.score(identifier)
    assert isinstance(actual, dict)
    assert _canonical_score(actual) == _canonical_score(_source_score(source))


@pytest.mark.parametrize("width", (1, 2, 3, 8))
@pytest.mark.parametrize("kind", ("register", "zero", "all_ones", "sparse"))
def test_complete_small_and_eight_bit_source_call_and_field_view_boundaries(
    width: int, kind: str
) -> None:
    from qualtran.bloqs.gf_arithmetic import GF2Addition, GF2AddK

    constant = {
        "register": None,
        "zero": 0,
        "all_ones": (1 << width) - 1,
        "sparse": ((1 << (width - 1)) | 1),
    }[kind]
    dtype = binary_field_encoding(width).register_type()
    ports = Signature.build_from_dtypes(
        **({"x": dtype, "y": dtype} if constant is None else {"x": dtype})
    )
    operation = "GF2Addition" if constant is None else "GF2AddK"
    parameters = {} if constant is None else {"k": constant}
    record = field_addition_catalog_program(operation, ports, parameters)
    assert record is not None
    source = GF2Addition(width) if constant is None else GF2AddK(width, constant)
    original = source.decompose_bloq()
    assert _native_trace(record) == _source_trace(SourceCalls, original)
    actual = field_addition_score(operation, ports, parameters)
    assert actual is not None
    assert _canonical_score(dict(actual)) == _canonical_score(_source_score(original))


@pytest.mark.parametrize("identifier", ("gf2_addition_symbolic", "gf2_add_k_symbolic"))
def test_symbolic_factories_retain_original_decomposition_refusal(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    from qualtran import DecomposeTypeError

    with pytest.raises(DecomposeTypeError):
        originals[1][identifier].decompose_bloq()
    assert native.decomposition(identifier) is None
