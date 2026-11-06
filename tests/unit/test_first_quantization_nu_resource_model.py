"""Compare momentum-transfer costs and live workspace with the locked source."""

import pytest

from flagquantum.algorithms.first_quantization_nu_resource_model import (
    source_prepare_nu_cost,
)
from flagquantum.compiler.registers.arithmetic_resource_model import (
    source_product_cost,
    source_sum_of_squares_cost,
)
from flagquantum.compiler.registers.ordering import source_greater_cost
from flagquantum.compiler.registers.uniform_preparation import (
    source_uniform_preparation_cost,
)
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("count", [2, 3, 5, 6, 8, 9, 14, 31, 32768])
def test_uniform_cost(count: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )

    assert_source_model(
        source_uniform_preparation_cost(count), PrepareUniformSuperposition(count)
    )


@pytest.mark.parametrize(
    ("left", "right"), [(1, 1), (1, 2), (2, 1), (2, 4), (5, 3), (4, 8), (8, 4)]
)
def test_momentum_arithmetic_costs(left: int, right: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.arithmetic import GreaterThan, Product, SumOfSquares

    assert_source_model(source_greater_cost(left, right), GreaterThan(left, right))
    for model, source in (
        (source_product_cost(left, right), Product(left, right)),
        (source_sum_of_squares_cost(left, right), SumOfSquares(left, right)),
    ):
        assert_source_model(model, source)
        with pytest.raises(DecomposeNotImplementedError):
            source.decompose_bloq()


@pytest.mark.parametrize(
    ("p", "m"), [(1, 3), (2, 2), (2, 5), (2, 64), (3, 9), (6, 32768)]
)
@pytest.mark.parametrize("adjoint", [False, True])
def test_prepare_nu_cost_and_bystanders(p: int, m: int, adjoint: bool) -> None:
    pytest.importorskip("qualtran")
    from qualtran import Adjoint, DecomposeNotImplementedError, LeftDangle
    from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu import (
        PrepareNuState,
    )
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )

    source = PrepareNuState(p, m)
    if adjoint:
        source = source.adjoint()
    assert_source_model(source_prepare_nu_cost(p, m, adjoint=adjoint), source)
    composite = source.decompose_bloq()
    model = source_prepare_nu_cost(p, m, adjoint=adjoint, decomposed=True)
    assert_source_model(model, composite)

    live = {edge for edge in composite.connections if edge.left.binst == LeftDangle}
    stages = {}
    for node, incoming, outgoing in composite.iter_bloqnections():
        live.difference_update(incoming)
        bloq = node.bloq.subbloq if isinstance(node.bloq, Adjoint) else node.bloq
        if isinstance(bloq, PrepareUniformSuperposition):
            assert len(incoming) == len(outgoing) == 1
            assert incoming[0].left.reg.name == outgoing[0].right.reg.name == "m"
            name = f"PrepareUniformSuperposition({m})"
        else:
            with pytest.raises(DecomposeNotImplementedError):
                node.bloq.decompose_bloq()
            name = type(bloq).__name__
        stages[name] = sum(edge.shape for edge in live)
        live.update(outgoing)
    assert stages == {
        call.component.name: call.bystander_qubits for call in model.calls
    }


def test_prepare_nu_source_empty_m_is_explicit() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu import (
        PrepareNuState,
    )
    from qualtran.resource_counting import QubitCount, get_cost_value

    with pytest.raises(RuntimeError, match="does not accept Soquets"):
        get_cost_value(PrepareNuState(2, 1), QubitCount())
    with pytest.raises(ValueError, match="m_param >= 2"):
        source_prepare_nu_cost(2, 1)
