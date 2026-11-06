"""Source parity for shape-only dense and sparse rotation resources."""

import pytest

from flagquantum.algorithms.primitives.symbolic_rotation_resource_model import (
    source_rotation_preparation_resource_formula,
    source_sparse_rotation_preparation_resource_formula,
)
from flagquantum.compiler.registers.symbolic_resource_model import (
    SymbolicResourceFormula,
)

pytestmark = pytest.mark.unit


def assert_source_cost(formula: SymbolicResourceFormula, source: object) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import simplify

    cost = get_cost_value(source, QECGatesCost())
    for field in ("and_bloq", "clifford", "measurement", "t", "toffoli"):
        assert simplify(getattr(formula, field) - getattr(cost, field)) == 0, field
    assert simplify(formula.logical_qubits - get_cost_value(source, QubitCount())) == 0


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("controls", [0, 1, 3])
def test_dense_symbolic_widths_and_bound_formulas(controls: int) -> None:
    pytest.importorskip("qualtran")
    sympy = pytest.importorskip("sympy")
    from qualtran.bloqs.state_preparation.state_preparation_via_rotation import (
        StatePreparationViaRotations,
    )
    from qualtran.symbolics import HasLength

    n, b = sympy.symbols("n b", positive=True, integer=True)
    dense = source_rotation_preparation_resource_formula(n, b, controls)
    assert_source_cost(dense, StatePreparationViaRotations(HasLength(n), b, controls))
    for count, phase_bits in ((1, 2), (4, 2), (8, 8)):
        bound = source_rotation_preparation_resource_formula(
            count, phase_bits, controls
        )
        assert_source_cost(
            bound, StatePreparationViaRotations(HasLength(count), phase_bits, controls)
        )
        for field in (
            "and_bloq",
            "clifford",
            "measurement",
            "toffoli",
            "logical_qubits",
        ):
            assert int(
                getattr(dense, field).subs({n: count, b: phase_bits})
            ) == getattr(bound, field)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("explicit_target", [False, True])
def test_sparse_symbolic_permutation_bound_and_width(explicit_target: bool) -> None:
    pytest.importorskip("qualtran")
    sympy = pytest.importorskip("sympy")
    from qualtran.bloqs.state_preparation.sparse_state_preparation_via_rotations import (
        SparseStatePreparationViaRotations,
    )
    from qualtran.symbolics import HasLength

    n, d, b = sympy.symbols("n d b", positive=True, integer=True)
    target = n if explicit_target else None
    source = (
        SparseStatePreparationViaRotations(
            HasLength(d), HasLength(d), n, b, target_bitsize=n
        )
        if explicit_target
        else SparseStatePreparationViaRotations.from_n_coeffs(n, d, b)
    )
    symbolic = source_sparse_rotation_preparation_resource_formula(
        n, d, b, target_width=target
    )
    assert_source_cost(symbolic, source)
    for size, nonzero, width in ((2, 1, 2), (3, 2, 2), (16, 3, 8)):
        bound = source_sparse_rotation_preparation_resource_formula(
            size, nonzero, width, target_width=size if explicit_target else None
        )
        concrete = (
            SparseStatePreparationViaRotations(
                HasLength(nonzero), HasLength(nonzero), size, width, target_bitsize=size
            )
            if explicit_target
            else SparseStatePreparationViaRotations.from_n_coeffs(size, nonzero, width)
        )
        assert_source_cost(bound, concrete)
        for field in (
            "and_bloq",
            "clifford",
            "measurement",
            "toffoli",
            "logical_qubits",
        ):
            assert int(
                getattr(symbolic, field).subs({n: size, d: nonzero, b: width})
            ) == getattr(bound, field)


def test_sparse_shape_requires_a_nonempty_target() -> None:
    with pytest.raises(ValueError, match="nonempty target"):
        source_sparse_rotation_preparation_resource_formula(1, 1, 2)


@pytest.mark.ftqc_reference
def test_sparse_concrete_width_rebuild_can_exceed_symbolic_fallback() -> None:
    pytest.importorskip("qualtran")
    sympy = pytest.importorskip("sympy")
    from qualtran.bloqs.state_preparation.sparse_state_preparation_via_rotations import (
        SparseStatePreparationViaRotations,
    )

    n = sympy.Symbol("n", positive=True, integer=True)
    symbolic = source_sparse_rotation_preparation_resource_formula(n, 1, 2)
    concrete = source_sparse_rotation_preparation_resource_formula(1024, 1, 2)
    assert int(symbolic.logical_qubits.subs(n, 1024)) == 19
    assert concrete.logical_qubits == 20
    assert_source_cost(
        concrete, SparseStatePreparationViaRotations.from_n_coeffs(1024, 1, 2)
    )
