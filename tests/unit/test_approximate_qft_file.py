"""Verify the locked approximate QFT declarations and retained cost boundary."""

from collections import Counter

import pytest
from sympy import Symbol, simplify

from examples.algorithms.approximate_qft_catalog import (
    approximate_qft_examples,
    approximate_qft_graph,
)
from flagquantum.algorithms.primitives.approximate_qft import approximate_qft_ir
from flagquantum.algorithms.primitives.approximate_qft_resource_model import (
    ApproximateQFTCall,
    source_approximate_qft_calls,
    source_approximate_qft_default_precision,
    source_approximate_qft_epsilon_precision,
    source_approximate_qft_formula,
    source_approximate_qft_signature,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.core.ir.program.registers import RegisterSide

pytestmark = pytest.mark.unit
_N, _P, _EPS = Symbol("n"), Symbol("p"), Symbol("epsilon")


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("width", "precision"),
    [
        (1, 0),
        (1, 3),
        (5, 1),
        (5, 3),
        (6, 5),
        (50, 25),
        (10, 4),
        (123, 7),
        (_N, _P),
        (_N, source_approximate_qft_epsilon_precision(_N, _EPS)),
    ],
)
@pytest.mark.parametrize("with_reverse", [False, True])
def test_locked_calls_signatures_and_costs(
    width: SymbolicInt, precision: SymbolicInt, with_reverse: bool
) -> None:
    source_package = pytest.importorskip("qualtran.bloqs.qft")
    costs = pytest.importorskip("qualtran.resource_counting")
    source = source_package.ApproximateQFT(width, precision, with_reverse=with_reverse)
    signature = source_approximate_qft_signature(width, precision)
    assert [(reg.name, reg.bitsize) for reg in signature] == [
        (reg.name, reg.bitsize) for reg in source.signature
    ]
    data, *gradient = signature
    assert all(reg.side == RegisterSide.THRU for reg in signature)
    assert data.dtype.encoding == "uint"
    if precision != 0:
        (port,) = gradient
        assert dict(port.dtype.parameters) == {
            "fractional_width": precision,
            "signed": False,
        }
    else:
        assert not gradient
    normalized: dict[ApproximateQFTCall, SymbolicInt] = {}
    for child, count in source.build_call_graph(costs.SympySymbolAllocator()).items():
        name = type(child).__name__
        if name == "AddIntoPhaseGrad":
            assert (child.controlled_by, child.right_shift, child.sign) == (1, 1, 1)
            normalized[(name, (child.x_bitsize, child.phase_bitsize))] = count
        else:
            assert name in ("Hadamard", "TwoBitSwap")
            normalized[(name, ())] = count
    assert (
        source_approximate_qft_calls(width, precision, with_reverse=with_reverse)
        == normalized
    )
    expected = costs.get_cost_value(source, costs.QECGatesCost())
    model = source_approximate_qft_formula(width, precision, with_reverse=with_reverse)
    for key in (
        "t",
        "toffoli",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
        "cswap",
    ):
        assert simplify(getattr(model, key) - getattr(expected, key)) == 0
    assert (
        simplify(
            model.logical_qubits - costs.get_cost_value(source, costs.QubitCount())
        )
        == 0
    )


@pytest.mark.ftqc_reference
def test_locked_precision_and_notebook_symbolic_t_expression() -> None:
    source = pytest.importorskip("qualtran.bloqs.qft").ApproximateQFT
    helper = pytest.importorskip("qualtran.resource_counting.t_counts_from_sigma")
    for width in (1, 10, 123, 1000, 2000, 10000, _N):
        assert (
            source_approximate_qft_default_precision(width)
            == source(width).phase_bitsize
        )
    for n in (1000, 2000, 10000):
        for bits in (1, 4, 7, 10):
            eps = 2.0**-bits
            assert (
                source_approximate_qft_epsilon_precision(n, eps)
                == source.from_epsilon(n, eps).phase_bitsize
            )
    original = source.from_epsilon(_N, _EPS)
    precision = source_approximate_qft_epsilon_precision(_N, _EPS)
    assert precision == original.phase_bitsize
    _, leaves = original.call_graph()
    toffoli = source_approximate_qft_formula(_N, precision).toffoli
    assert simplify(helper.t_counts_from_sigma(leaves) - 4 * toffoli) == 0


def test_raw_negative_estimate_does_not_change_emitted_zero_window() -> None:
    assert source_approximate_qft_formula(5, 1).toffoli == -8
    for adjoint in (False, True):
        counts = Counter(
            gate.name
            for gate in approximate_qft_ir(
                tuple(range(5)), (5,), (), (), adjoint=adjoint
            )
        )
        assert counts == {"h": 5, "swap": 2}


def test_notebook_examples_and_concrete_one_level_graph() -> None:
    assert approximate_qft_examples() == ((6, 5), (50, 25))
    assert approximate_qft_graph(6, 5).leaf_counts == {
        "AddIntoPhaseGrad(1, 2; shift 1)": 1,
        "AddIntoPhaseGrad(2, 3; shift 1)": 1,
        "AddIntoPhaseGrad(3, 4; shift 1)": 1,
        "AddIntoPhaseGrad(4, 5; shift 1)": 2,
        "Hadamard": 6,
        "TwoBitSwap": 3,
    }
