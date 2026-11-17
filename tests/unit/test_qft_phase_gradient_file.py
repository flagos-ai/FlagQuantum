"""Compare the locked phase-gradient Fourier signature, graph and model."""

import pytest
from sympy import Symbol, simplify

from flagquantum.algorithms.primitives.qft_phase_gradient_resource_model import (
    PhaseGradientQFTCall,
    PhaseGradientQFTNode,
    ProductAccumulationCall,
    source_phase_gradient_qft_calls,
    source_phase_gradient_qft_formula,
    source_phase_gradient_qft_signature,
)
from flagquantum.compiler._component_trace.resource_generalization import CliffordCall
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.core.ir.program.registers import RegisterSide

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 10, 123, Symbol("n")])
@pytest.mark.parametrize("with_reverse", [False, True])
def test_locked_phase_gradient_qft_declarations(
    width: SymbolicInt, with_reverse: bool
) -> None:
    source_package = pytest.importorskip("qualtran.bloqs.qft")
    costs = pytest.importorskip("qualtran.resource_counting")
    source = source_package.QFTPhaseGradient(width, with_reverse=with_reverse)
    signature = source_phase_gradient_qft_signature(width)
    assert [(reg.name, reg.bitsize) for reg in signature] == [
        (reg.name, reg.bitsize) for reg in source.signature
    ]
    data, gradient = signature
    assert data.side == gradient.side == RegisterSide.THRU
    assert data.dtype.encoding == "uint"
    assert dict(gradient.dtype.parameters) == {
        "fractional_width": width,
        "signed": False,
    }
    assert source.signature[1].dtype.num_frac == width

    normalized: dict[PhaseGradientQFTNode, SymbolicInt] = {}
    for child, count in source.build_call_graph(costs.SympySymbolAllocator()).items():
        name = type(child).__name__
        if name == "QFTPhaseGradient":
            node: PhaseGradientQFTNode = PhaseGradientQFTCall(
                child.bitsize, child.with_reverse
            )
        elif name == "PlusEqualProduct":
            assert not child.is_adjoint
            node = ProductAccumulationCall(
                child.a_bitsize, child.b_bitsize, child.result_bitsize
            )
        else:
            node = (
                CliffordCall("h", 1)
                if name == "Hadamard"
                else CliffordCall("swap_bit", 2)
            )
        normalized[node] = count
    assert (
        source_phase_gradient_qft_calls(PhaseGradientQFTCall(width, with_reverse))
        == normalized
    )
    expected = costs.get_cost_value(source, costs.QECGatesCost())
    model = source_phase_gradient_qft_formula(width, with_reverse=with_reverse)
    for field in (
        "t",
        "clifford",
        "rotation",
        "measurement",
        "and_bloq",
        "cswap",
        "toffoli",
    ):
        assert simplify(getattr(model, field) - getattr(expected, field)) == 0
    assert (
        simplify(
            model.logical_qubits - costs.get_cost_value(source, costs.QubitCount())
        )
        == 0
    )


def test_source_duplicate_child_is_retained_as_one_model_call() -> None:
    root = PhaseGradientQFTCall(4, False)
    assert source_phase_gradient_qft_calls(root) == {
        PhaseGradientQFTCall(2, True): 1,
        ProductAccumulationCall(2, 2, 4): 1,
    }
    assert source_phase_gradient_qft_formula(4, with_reverse=False).t == 40
