"""Source text-symbol widths retain their atomic names without Python evaluation."""

import pytest
import sympy

from flagquantum.compiler.formats._legacy_expression import parse_legacy_expression
from flagquantum.compiler.formats.legacy_arguments import (
    deserialize_integer_argument,
    serialize_integer_argument,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("name", (r"b_\text{grad}", r"\text{phase_bitsize}"))
def test_atomic_names_and_arithmetic_preserve_existing_symbol_identity(
    name: str,
) -> None:
    symbol = sympy.Symbol(name)
    assert parse_legacy_expression(name) == symbol
    assert deserialize_integer_argument(serialize_integer_argument(symbol)) == symbol
    expression = 2 * symbol + sympy.Symbol("fq_text_symbol_0")
    assert (
        deserialize_integer_argument(serialize_integer_argument(expression))
        == expression
    )


def test_original_symbolic_phase_width_library_roundtrips_without_source_edits() -> (
    None
):
    from qualtran.bloqs.state_preparation.state_preparation_via_rotation import (
        _state_prep_via_rotation_symb_phasegrad,
    )
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = _state_prep_via_rotation_symb_phasegrad.make()
    assert bloqs_from_proto(bloqs_to_proto(source, max_depth=0))[0] == source
    for original in (source, source.adjoint()):
        library = _library(original, 0)
        encoded = serialize_component_library(library)
        assert (
            serialize_component_library(deserialize_component_library(encoded))
            == encoded
        )
        ports = library.entries[0].declaration.registers
        assert ports is not None
        width = ports[-1].bitsize
        assert width == sympy.Symbol(str(source.phase_bitsize))


@pytest.mark.parametrize(
    "text",
    (
        r"b_\text{grad}.__class__",
        r"b_\text{grad}()",
        r"\text{__import__('os')}",
        r"b_\text{grad}_suffix",
        r"prefix\text{grad}",
        r"\text{grad}[0]",
        r"\text{grad}; __import__('os')",
        r"2**20000 + b_\text{grad}",
    ),
)
def test_text_names_do_not_admit_calls_attributes_indexing_or_budget_bypasses(
    text: str,
) -> None:
    with pytest.raises(ValueError):
        parse_legacy_expression(text)
