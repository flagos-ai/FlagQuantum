"""Compare the locked coefficient words and opaque inverse-square-root costs."""

import pytest

from flagquantum.algorithms.primitives.grid_inverse_sqrt import (
    grid_inverse_sqrt_qrom_data,
    source_grid_inverse_sqrt_newton_cost,
    source_grid_inverse_sqrt_newton_signature,
    source_grid_inverse_sqrt_polynomial_cost,
    source_grid_inverse_sqrt_polynomial_signature,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "address,precision", [(2, 4), (3, 32), (6, 15), (7, 12), (8, 30), (5, 55)]
)
def test_source_inverse_sqrt_coefficient_tables(address, precision):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt import (
        build_qrom_data_for_poly_fit,
        get_inverse_square_root_poly_coeffs,
    )

    source = build_qrom_data_for_poly_fit(
        address, precision, get_inverse_square_root_poly_coeffs()
    )
    assert grid_inverse_sqrt_qrom_data(address, precision) == tuple(
        tuple(int(word) for word in bank) for bank in source
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("widths", [(4, 8, 12), (5, 15, 22), (7, 8, 4), (7, 8, 12)])
def test_source_inverse_sqrt_resource_graphs(widths):
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError, QAny, Side
    from qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt import (
        NewtonRaphsonApproxInverseSquareRoot,
        PolynmomialEvaluationInverseSquareRoot,
    )
    from qualtran.resource_counting import (
        QECGatesCost,
        QubitCount,
        SympySymbolAllocator,
        get_cost_value,
    )

    for native, signature, original in (
        (
            source_grid_inverse_sqrt_newton_cost(*widths),
            source_grid_inverse_sqrt_newton_signature(*widths),
            NewtonRaphsonApproxInverseSquareRoot(*widths),
        ),
        (
            source_grid_inverse_sqrt_polynomial_cost(*widths),
            source_grid_inverse_sqrt_polynomial_signature(*widths),
            PolynmomialEvaluationInverseSquareRoot(*widths),
        ),
    ):
        assert all(isinstance(register.dtype, QAny) for register in original.signature)
        assert all(register.side == Side.THRU for register in original.signature)
        assert all(register.dtype.encoding == "opaque" for register in signature)
        assert signature.thru_registers_only
        assert [
            (register.name, register.bitsize, register.shape) for register in signature
        ] == [
            (register.name, register.bitsize, register.shape)
            for register in original.signature
        ]
        assert [register.name for register in signature.lefts()] == [
            register.name for register in signature.rights()
        ]
        with pytest.raises(DecomposeNotImplementedError):
            original.decompose_bloq()
        calls = original.build_call_graph(SympySymbolAllocator())
        assert [(edge.component.name, edge.repetitions) for edge in native.calls] == [
            (type(child).__name__, count) for child, count in calls.items()
        ]
        for model, source in (
            (native, original),
            *(
                (edge.component, child)
                for edge, child in zip(native.calls, calls, strict=True)
            ),
        ):
            actual = estimate_component_resources(model)
            expected = get_cost_value(source, QECGatesCost())
            for field in (
                "t",
                "toffoli",
                "cswap",
                "and_bloq",
                "clifford",
                "rotation",
                "measurement",
            ):
                assert getattr(actual.gates, field) == getattr(expected, field)
            assert actual.peak_logical_qubits == get_cost_value(source, QubitCount())
