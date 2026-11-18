"""Whole source signature, concrete graph and symbolic sorting model parity."""

from dataclasses import asdict
from hashlib import sha1
from pathlib import Path

import pytest

from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.sorting.in_place_sort_declaration import (
    in_place_sort_signature,
    source_in_place_sort_cost,
    source_in_place_sort_resource_formula,
)
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
    signed_register_type,
    unsigned_register_type,
)
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_frozen_complete_file_and_documentation_spec() -> None:
    from qualtran import DecomposeNotImplementedError, QUInt
    from qualtran.bloqs.arithmetic.lists import sort_in_place

    raw = Path(sort_in_place.__file__).read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "d4da3727ad3ce11294c140870f2dbf4e1cb0f9fd"
    )
    assert sort_in_place._SORT_IN_PLACE_DOC.examples == ()
    with pytest.raises(DecomposeNotImplementedError):
        sort_in_place.SortInPlace(4, QUInt(3)).decompose_bloq()


@pytest.mark.parametrize("length", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("width", [1, 2, 3, 5])
def test_complete_concrete_direct_calls_categories_and_width(
    length: int, width: int
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Xor
    from qualtran.bloqs.arithmetic.lists import SortInPlace
    from qualtran.bloqs.arithmetic.sorting import Comparator
    from qualtran.resource_counting import SympySymbolAllocator

    source = SortInPlace(length, QUInt(width))
    model = source_in_place_sort_cost(length, width)
    calls = source.build_call_graph(SympySymbolAllocator())
    assert list(calls) == [
        Comparator(width),
        Comparator(width).adjoint(),
        Xor(QUInt(width)),
    ]
    assert (
        list(calls.values())
        == [edge.repetitions for edge in model.calls]
        == [3 * length**2] * 3
    )
    assert_source_model(model, source)
    formula = source_in_place_sort_resource_formula(length, width)
    report = estimate_component_resources(model)
    assert formula.logical_qubits == report.peak_logical_qubits
    counts = asdict(formula)
    del counts["logical_qubits"]
    assert counts == asdict(report.gates)


def test_full_signatures_and_symbolic_cost_queries() -> None:
    import sympy
    from qualtran import QAny, QInt, QUInt
    from qualtran.bloqs.arithmetic.lists import SortInPlace
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    length, width = sympy.symbols("l n", positive=True, integer=True)
    for count, bits in ((length, width), (length, 3), (4, width)):
        for dtype, native_dtype in (
            (QUInt(bits), unsigned_register_type(bits)),
            (QInt(bits), signed_register_type(bits)),
            (QAny(bits), opaque_register_type(bits)),
        ):
            source = SortInPlace(count, dtype)
            signature = in_place_sort_signature(count, native_dtype)
            assert [
                (r.name, r.dtype, r.shape_symbolic, r.side.name) for r in signature
            ] == [
                ("xs", native_dtype, (count,), "THRU"),
                (
                    "pi",
                    bounded_unsigned_register_type(
                        source.index_dtype.num_qubits, count
                    ),
                    (count,),
                    "THRU",
                ),
            ]
            assert [
                (r.name, r.dtype, r.shape_symbolic, r.side.name)
                for r in source.signature
            ] == [
                ("xs", dtype, (count,), "THRU"),
                ("pi", source.index_dtype, (count,), "THRU"),
            ]
            assert all(r.side.name == "THRU" for r in signature)
            formula = source_in_place_sort_resource_formula(count, bits)
            observed = asdict(formula)
            peak = observed.pop("logical_qubits")
            expected = get_cost_value(source, QECGatesCost()).asdict()
            for category in observed.keys() | expected.keys():
                assert (
                    sympy.simplify(
                        observed.get(category, 0) - expected.get(category, 0)
                    )
                    == 0
                )
            assert sympy.simplify(peak - get_cost_value(source, QubitCount())) == 0


@pytest.mark.parametrize("length", [0, -1, True, 1.5])
def test_model_rejects_invalid_lengths(length: int) -> None:
    with pytest.raises(ValueError, match="length"):
        source_in_place_sort_cost(length, 2)
    with pytest.raises(ValueError, match="length"):
        source_in_place_sort_resource_formula(length, 2)
