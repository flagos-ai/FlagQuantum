"""Price inverse k-XOR entry paths using the existing exact adder polarity."""

import pytest
import sympy
from qualtran import DecomposeTypeError
from qualtran.bloqs.optimization.k_xor_sat import kikuchi_adjacency_matrix as source
from qualtran.resource_counting import QECGatesCost, get_cost_value

from flagquantum.algorithms.kikuchi_matrix import (
    example_kikuchi_matrix_entry,
    symbolic_kikuchi_matrix_entry,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("symbolic", (False, True))
@pytest.mark.parametrize("controlled", (False, True))
@pytest.mark.parametrize("adjoint", (False, True))
def test_complete_forward_inverse_rhs_and_matrix_categories(
    symbolic: bool, controlled: bool, adjoint: bool
) -> None:
    native = (
        symbolic_kikuchi_matrix_entry() if symbolic else example_kikuchi_matrix_entry()
    )
    original = (
        source._kikuchi_matrix_entry_symb()
        if symbolic
        else source._kikuchi_matrix_entry()
    )
    rhs = native.rhs(controlled=controlled)
    from qualtran.bloqs.optimization.k_xor_sat.load_kxor_instance import (
        PRGAUniqueConstraintRHS,
    )

    source_rhs = PRGAUniqueConstraintRHS(
        original.inst, original.entry_bitsize, controlled
    )
    matrix = original.controlled() if controlled else original
    pairs = [
        (
            rhs.resource_gates(adjoint=adjoint),
            source_rhs.adjoint() if adjoint else source_rhs,
        )
    ]
    if controlled and adjoint:
        with pytest.raises(DecomposeTypeError, match="atomic"):
            get_cost_value(matrix.adjoint(), QECGatesCost())
        with pytest.raises(NotImplementedError, match="atomic"):
            native.resource_gates(controlled=True, adjoint=True)
    else:
        pairs.append(
            (
                native.resource_gates(controlled=controlled, adjoint=adjoint),
                matrix.adjoint() if adjoint else matrix,
            )
        )
    for actual, expected in pairs:
        model = get_cost_value(expected, QECGatesCost())
        for field in (
            "t",
            "toffoli",
            "cswap",
            "and_bloq",
            "clifford",
            "rotation",
            "measurement",
        ):
            assert sympy.simplify(getattr(actual, field) - getattr(model, field)) == 0
    assert (
        rhs.resource_gates().clifford - rhs.resource_gates(adjoint=True).clifford == 2
    )
