"""Full duplicate model source calls, symbolic lifetimes and failed peaks."""

from dataclasses import asdict

import pytest

from flagquantum.compiler.registers.sorting.sorted_duplicates_resource_model import (
    source_sorted_duplicates_cost,
    source_sorted_duplicates_gate_cost,
    source_sorted_duplicates_resource_formula,
)
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("length", [2, 3, 4])
@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("controlled", [False, True])
def test_concrete_full_model_matches_each_source_category_and_peak(
    length: int, width: int, signed: bool, controlled: bool
) -> None:
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic.lists import HasDuplicates
    from qualtran.resource_counting import SympySymbolAllocator

    source = HasDuplicates(length, (QInt if signed else QUInt)(width), controlled)
    native = source_sorted_duplicates_cost(
        length, width, signed=signed, controlled=controlled
    )
    assert [n for n in source.build_call_graph(SympySymbolAllocator()).values()] == [
        c.repetitions for c in native.calls
    ]
    assert_source_model(native, source)


def test_actual_unsigned_symbolic_factories_and_controlled_metadata() -> None:
    import sympy
    from qualtran import QInt
    from qualtran.bloqs.arithmetic.lists import HasDuplicates
    from qualtran.bloqs.arithmetic.lists.has_duplicates import (
        _has_duplicates,
        _has_duplicates_symb,
        _has_duplicates_symb_len,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    for example in (_has_duplicates, _has_duplicates_symb, _has_duplicates_symb_len):
        base = example.make()
        for controlled in (False, True):
            source = HasDuplicates(base.l, base.dtype, controlled)
            formula = source_sorted_duplicates_resource_formula(
                source.l, source.dtype.bitsize, controlled=controlled
            )
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
    # The controlled singleton has no half-comparison child, even for signed symbols.
    width = sympy.Symbol("w")
    singleton = HasDuplicates(1, QInt(width), True)
    assert source_sorted_duplicates_resource_formula(
        1, width, signed=True, controlled=True
    ).logical_qubits == get_cost_value(singleton, QubitCount())


def test_failed_source_peaks_keep_their_independent_qec_queries() -> None:
    import sympy
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic.lists import HasDuplicates
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    for length, width, signed in ((1, 2, False), (4, sympy.Symbol("n"), True)):
        source = HasDuplicates(length, (QInt if signed else QUInt)(width))
        gates = source_sorted_duplicates_gate_cost(length, width, signed=signed)
        assert gates.asdict() == get_cost_value(source, QECGatesCost()).asdict()
        with pytest.raises(RuntimeError):
            get_cost_value(source, QubitCount())
        with pytest.raises(ValueError, match="Original"):
            source_sorted_duplicates_resource_formula(length, width, signed=signed)


def test_half_dependency_reuses_out_of_place_addition_categories_and_peak() -> None:
    import sympy
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic import LinearDepthHalfLessThan
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler.registers.comparison.half_ordering import (
        source_half_comparison_gate_cost,
        source_half_comparison_resource_formula,
    )

    for width in (1, 2, 3, sympy.Symbol("w")):
        for signed in (False, True):
            for adjoint in (False, True):
                source = LinearDepthHalfLessThan(
                    (QInt if signed else QUInt)(width), uncompute=adjoint
                )
                gates = source_half_comparison_gate_cost(
                    width, signed=signed, adjoint=adjoint
                )
                assert gates.asdict() == get_cost_value(source, QECGatesCost()).asdict()
                if signed and type(width) is not int:
                    with pytest.raises(ValueError, match="SignExtend"):
                        source_half_comparison_resource_formula(
                            width, signed=signed, adjoint=adjoint
                        )
                else:
                    formula = source_half_comparison_resource_formula(
                        width, signed=signed, adjoint=adjoint
                    )
                    assert (
                        sympy.simplify(
                            formula.logical_qubits
                            - get_cost_value(source, QubitCount())
                        )
                        == 0
                    )
