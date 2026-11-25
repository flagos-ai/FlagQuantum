"""Whole set-difference factories, direct model and reusable dependency parity."""

from dataclasses import asdict
from hashlib import sha1
from pathlib import Path

import pytest

from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.comparison.compare_exchange import (
    source_compare_exchange_cost,
)
from flagquantum.compiler.registers.comparison.equality import (
    source_equality_constant_cost,
    source_equality_cost,
)
from flagquantum.compiler.registers.control.multi_control import (
    source_multi_control_x_cost,
)
from flagquantum.compiler.registers.sorting.bitonic_merge import (
    source_bitonic_merge_cost,
)
from flagquantum.compiler.registers.sorting.symmetric_difference_declaration import (
    source_symmetric_difference_cost,
    symmetric_difference_signature,
)
from flagquantum.core.ir.program.register_types import unsigned_register_type
from tests.unit.source_example_protocols import source_example_protocols
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "name,symbolic",
    (
        ("_symm_diff", False),
        ("_symm_diff_symb", True),
        ("_symm_diff_equal_size_symb", True),
    ),
)
def test_actual_factory_raw_protocols(name: str, symbolic: bool) -> None:
    from qualtran.bloqs.arithmetic.lists import symmetric_difference

    observed = source_example_protocols(getattr(symmetric_difference, name))
    assert {mode: status for mode, (status, _) in observed.items()} == {
        "make": "PASS",
        "decompose": "NA" if symbolic else "MISSING",
        "counts": "UNVERIFIED",
        "serialize": "FAIL" if symbolic else "PASS",
        "qtyping": "NA",
    }


@pytest.mark.parametrize("equal_size", (False, True))
def test_original_symbolic_t_bound_with_native_formula(equal_size: bool) -> None:
    import sympy
    from qualtran.bloqs.arithmetic.lists.symmetric_difference import (
        _symm_diff_equal_size_symb,
        _symm_diff_symb,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.compiler._component_trace.gate_budget import t_and_ccz_budget
    from flagquantum.compiler._component_trace.symbolic_math import (
        asymptotic_order,
        ceil,
        log2,
    )
    from flagquantum.compiler.registers.sorting.symmetric_difference_declaration import (
        source_symmetric_difference_resource_formula,
    )

    source = (_symm_diff_equal_size_symb if equal_size else _symm_diff_symb).make()
    formula = source_symmetric_difference_resource_formula(
        source.n_lhs, source.n_rhs, source.n_diff, source.dtype.num_qubits
    )
    budget = t_and_ccz_budget(formula)
    total = budget["n_t"] + 4 * budget["n_ccz"]
    assert (
        sympy.simplify(total - get_cost_value(source, QECGatesCost()).total_t_count())
        == 0
    )
    bound = source.n_lhs * source.dtype.num_qubits * ceil(log2(source.n_lhs)) ** 2
    assert total in asymptotic_order(bound)


def test_complete_frozen_implementation_and_test_files() -> None:
    from qualtran.bloqs.arithmetic.lists import symmetric_difference

    for name, blob in (
        ("symmetric_difference.py", "72d13f822bddfe699e843b4b63a33c64f1493ede"),
        ("symmetric_difference_test.py", "0ad5a356b6aff8d4e25dbd2a15f6dcaf6af7bef8"),
    ):
        raw = (Path(symmetric_difference.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def test_actual_three_factories_full_signature_and_documented_subset() -> None:
    from qualtran.bloqs.arithmetic.lists.symmetric_difference import (
        _SYMMETRIC_DIFFERENCE_DOC,
        _symm_diff,
        _symm_diff_equal_size_symb,
        _symm_diff_symb,
    )

    assert _SYMMETRIC_DIFFERENCE_DOC.examples == (_symm_diff,)
    for example in (_symm_diff, _symm_diff_symb, _symm_diff_equal_size_symb):
        source = example.make()
        dtype = unsigned_register_type(source.dtype.num_qubits)
        signature = symmetric_difference_signature(
            source.n_lhs, source.n_rhs, source.n_diff, dtype
        )
        assert [
            (r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in signature
        ] == [
            (r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in source.signature
        ]
        assert all(r.dtype == dtype for r in tuple(signature)[:3])


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5])
def test_reusable_comparator_equality_and_merge_models(width: int) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Equals, EqualsAConstant
    from qualtran.bloqs.arithmetic.sorting import BitonicMerge, Comparator
    from qualtran.bloqs.mcmt import MultiControlX

    for adjoint in (False, True):
        compare = Comparator(width)
        assert_source_model(
            source_compare_exchange_cost(width, adjoint=adjoint),
            compare.adjoint() if adjoint else compare,
        )
        for size in (1, 2, 4):
            merge = BitonicMerge(size, width)
            assert_source_model(
                source_bitonic_merge_cost(size, width, adjoint=adjoint),
                merge.adjoint() if adjoint else merge,
            )
    assert_source_model(source_equality_cost(width), Equals(QUInt(width)))
    for value in (0, (1 << width) - 1, (1 << width) // 2):
        assert_source_model(
            source_equality_constant_cost(width, value), EqualsAConstant(width, value)
        )
        values = tuple((value >> i) & 1 for i in reversed(range(width)))
        assert_source_model(source_multi_control_x_cost(values), MultiControlX(values))


@pytest.mark.parametrize(
    "left,right,output,width",
    [
        (2, 1, 1, 2),
        (2, 1, 2, 2),
        (2, 2, 0, 1),
        (2, 2, 2, 2),
        (4, 2, 4, 4),
        (4, 4, 4, 3),
    ],
)
def test_complete_direct_graph_and_root_cost(
    left: int, right: int, output: int, width: int
) -> None:
    from qualtran import DecomposeNotImplementedError, QUInt
    from qualtran.bloqs.arithmetic.lists import SymmetricDifference
    from qualtran.resource_counting import (
        QECGatesCost,
        SympySymbolAllocator,
        get_cost_value,
    )

    source = SymmetricDifference(left, right, output, QUInt(width))
    native = source_symmetric_difference_cost(left, right, output, width)
    calls = source.build_call_graph(SympySymbolAllocator())
    assert [n for n in calls.values()] == [c.repetitions for c in native.calls]
    assert len(calls) == len(native.calls) == 10
    assert_source_model(native, source)
    for (child, _), edge in zip(calls.items(), native.calls, strict=True):
        observed = asdict(estimate_component_resources(edge.component).gates)
        expected = get_cost_value(child, QECGatesCost()).asdict()
        assert {k: v for k, v in observed.items() if v} == expected
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()


def test_actual_example_model_has_no_emitted_primitive_claim() -> None:
    from flagquantum.qec.resources.estimation import estimate_ftqc_resources

    report = estimate_ftqc_resources(source_symmetric_difference_cost(4, 2, 4, 4))
    assert report.emitted is None
    assert report.logical.gates.cswap == 96
    assert report.logical.gates.and_bloq == 205
    assert report.logical.gates.clifford == 2276
    assert report.logical.gates.measurement == 205
    assert report.logical.peak_logical_qubits == 55


def test_original_one_adjacency_peak_failure_and_separate_native_model() -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.lists import SymmetricDifference
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = SymmetricDifference(1, 1, 0, QUInt(1))
    native = estimate_component_resources(source_symmetric_difference_cost(1, 1, 0, 1))
    assert {k: v for k, v in asdict(native.gates).items() if v} == get_cost_value(
        source, QECGatesCost()
    ).asdict()
    with pytest.raises(RuntimeError):
        get_cost_value(source, QubitCount())
    assert native.peak_logical_qubits == 5


def test_all_symbolic_factory_models_and_mixed_binding() -> None:
    import sympy
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.lists import SymmetricDifference
    from qualtran.bloqs.arithmetic.lists.symmetric_difference import (
        _symm_diff_equal_size_symb,
        _symm_diff_symb,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler.registers.sorting.symmetric_difference_declaration import (
        source_symmetric_difference_resource_formula,
    )

    length, width, output = sympy.symbols("l w d", positive=True, integer=True)
    cases = (
        _symm_diff_symb.make(),
        _symm_diff_equal_size_symb.make(),
        SymmetricDifference(length, 2, output, QUInt(3)),
        SymmetricDifference(4, 2, 4, QUInt(width)),
        SymmetricDifference(4, 2, output, QUInt(3)),
        SymmetricDifference(3, 2, 3, QUInt(3)),
    )
    for source in cases:
        formula = source_symmetric_difference_resource_formula(
            source.n_lhs, source.n_rhs, source.n_diff, source.dtype.num_qubits
        )
        observed = asdict(formula)
        peak = observed.pop("logical_qubits")
        expected = get_cost_value(source, QECGatesCost()).asdict()
        for category in observed.keys() | expected.keys():
            assert (
                sympy.simplify(observed.get(category, 0) - expected.get(category, 0))
                == 0
            )
        expected_peak = get_cost_value(source, QubitCount())
        assert sympy.simplify(peak - expected_peak) == 0


def test_symbolic_dependency_formulas_preserve_unknown_control_and_binding() -> None:
    import sympy
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Equals, EqualsAConstant
    from qualtran.bloqs.arithmetic.sorting import BitonicMerge, Comparator
    from qualtran.bloqs.mcmt import MultiControlX
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.symbolics import HasLength

    from flagquantum.compiler.registers.comparison.compare_exchange import (
        source_compare_exchange_resource_formula,
    )
    from flagquantum.compiler.registers.comparison.equality import (
        source_equality_constant_resource_formula,
        source_equality_resource_formula,
    )
    from flagquantum.compiler.registers.control.multi_control import (
        source_multi_control_x_resource_formula,
    )
    from flagquantum.compiler.registers.sorting.bitonic_merge import (
        source_bitonic_merge_resource_formula,
    )

    length, width, value = sympy.symbols("l w v", positive=True, integer=True)
    cases = [
        (
            source_multi_control_x_resource_formula(width),
            MultiControlX(HasLength(width)),
        ),
        (source_multi_control_x_resource_formula(3), MultiControlX(HasLength(3))),
        (source_equality_resource_formula(width), Equals(QUInt(width))),
        (
            source_equality_constant_resource_formula(width, value),
            EqualsAConstant(width, value),
        ),
        (
            source_equality_constant_resource_formula(3, value),
            EqualsAConstant(3, value),
        ),
        (source_equality_constant_resource_formula(3, 2), EqualsAConstant(3, 2)),
        (source_compare_exchange_resource_formula(width), Comparator(width)),
        (source_bitonic_merge_resource_formula(4, width), BitonicMerge(4, width)),
        (source_bitonic_merge_resource_formula(length, 3), BitonicMerge(length, 3)),
        (source_bitonic_merge_resource_formula(3, 3), BitonicMerge(3, 3)),
    ]
    for pattern in ((0,), (1,), (0, 1), (0, 0, 1)):
        cases.append(
            (source_multi_control_x_resource_formula(pattern), MultiControlX(pattern))
        )
    for formula, source in cases:
        counts = asdict(formula)
        peak = counts.pop("logical_qubits")
        expected = get_cost_value(source, QECGatesCost()).asdict()
        for category in counts.keys() | expected.keys():
            assert (
                sympy.simplify(counts.get(category, 0) - expected.get(category, 0)) == 0
            )
        assert sympy.simplify(peak - get_cost_value(source, QubitCount())) == 0
