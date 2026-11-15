"""Check the pinned abstract Clifford leaf and both approximate-swap call graphs."""

import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler._component_trace.resource_decomposition import (
    decomposed_resource_callees,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    ControlledSwapCall,
    PhaseCall,
    arbitrary_clifford_cost_key,
    arbitrary_clifford_signature,
)
from flagquantum.compiler._component_trace.resource_width import (
    query_resource_logical_width,
)
from flagquantum.compiler.registers.layout.int_vector import int_vector_cost_key

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [0, 1, 2, 8, "unconstrained", None])
def test_arbitrary_clifford_cost_and_width_match_source(
    width: int | str | None,
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError, Side
    from qualtran.bloqs.bookkeeping import ArbitraryClifford
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import Symbol

    if width is None:
        bits = Symbol("n", positive=True, integer=True)
    elif isinstance(width, str):
        bits = Symbol("n")
    else:
        bits = width
    leaf = ArbitraryClifford(bits)
    assert leaf.signature[0].side == Side.THRU
    assert int_vector_cost_key(bits) == arbitrary_clifford_cost_key(bits)
    for adjoint in (False, True):
        source = leaf.adjoint() if adjoint else leaf
        key = arbitrary_clifford_cost_key(bits, adjoint=adjoint)
        assert key.width == get_cost_value(source, QubitCount()) == bits
        assert str(key) == str(source)
        (port,) = arbitrary_clifford_signature(key)
        original = source.signature[0]
        assert (port.name, port.bitsize, port.side.value, port.shape) == (
            original.name,
            original.bitsize,
            original.side.value,
            original.shape,
        )
        assert query_resource_logical_width(key) == bits
        assert source.call_graph(max_depth=1)[1] == {source: 1}
        with pytest.raises(NotImplementedError, match="declared callees"):
            tuple(decomposed_resource_callees(key, None))
        assert query_qec_gate_cost(key) == QECGateCost(clifford=1)
        assert get_cost_value(source, QECGatesCost()).asdict() == {"clifford": 1}
    with pytest.raises(DecomposeNotImplementedError):
        leaf.decompose_bloq()
    with pytest.raises(NotImplementedError, match="tensor simulation"):
        leaf.tensor_contract()


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 4, None])
@pytest.mark.parametrize("adjoint", [False, True])
def test_approximate_swap_keeps_distinct_source_clifford_widths_and_weights(
    width: int | None, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import TGate
    from qualtran.bloqs.bookkeeping import ArbitraryClifford
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from sympy import Symbol

    bits = Symbol("n", positive=True, integer=True) if width is None else width
    node = ControlledSwapCall(bits, approximate=True, adjoint=adjoint)
    source = CSwapApprox(bits)
    source = source.adjoint() if adjoint else source
    cliffords = (ArbitraryClifford(1), ArbitraryClifford(2))
    keys = (
        PhaseCall("t", adjoint),
        *(arbitrary_clifford_cost_key(leaf.n, adjoint=adjoint) for leaf in cliffords),
    )
    source_leaves = (
        TGate(is_adjoint=adjoint),
        *(leaf.adjoint() if adjoint else leaf for leaf in cliffords),
    )
    weights = source.call_graph()[1]
    assert dict(decomposed_resource_callees(node, None)) == {
        key: weights[leaf] for key, leaf in zip(keys, source_leaves, strict=True)
    }
    assert {k: v for k, v in asdict(query_qec_gate_cost(node)).items() if v != 0} == (
        get_cost_value(source, QECGatesCost()).asdict()
    )


def test_arbitrary_clifford_rejects_negative_width() -> None:
    with pytest.raises(ValueError, match="nonnegative integer width"):
        arbitrary_clifford_cost_key(-1)


def test_arbitrary_clifford_resource_owner_imports_in_a_fresh_process() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from flagquantum.compiler._component_trace.resource_generalization "
            "import arbitrary_clifford_cost_key",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        timeout=30,
    )
