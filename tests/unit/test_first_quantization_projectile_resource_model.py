"""Compare the projectile cost graphs against the locked source."""

from collections import Counter
from dataclasses import asdict

import pytest

from flagquantum.algorithms.first_quantization_nu_resource_model import (
    source_projectile_prepare_mu_cost,
    source_projectile_prepare_nu_adjoint_gate_cost,
    source_projectile_prepare_nu_cost,
)
from flagquantum.algorithms.first_quantization_projectile_resource_model import (
    source_projectile_power_two_cost,
    source_projectile_prepare_t_cost,
    source_projectile_select_t_cost,
    source_projectile_select_uv_cost,
)
from flagquantum.compiler._component_trace.call_graph import modeled_call_graph
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("p", "n", "eta", "precision"),
    [(1, 1, 2, 1), (2, 2, 3, 4), (2, 4, 5, 6), (6, 8, 10, 8)],
)
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.ftqc_reference
def test_projectile_kinetic_prepare(
    p: int, n: int, eta: int, precision: int, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.basic_gates import Toffoli
    from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_t import (
        PreparePowerTwoStateWithProj,
        PrepareTFirstQuantizationWithProj,
    )

    power = PreparePowerTwoStateWithProj(n, p, is_adjoint=adjoint)
    source = PrepareTFirstQuantizationWithProj(p, n, eta, precision, is_adjoint=adjoint)
    model = source_projectile_prepare_t_cost(p, n, adjoint=adjoint)
    assert_source_model(source_projectile_power_two_cost(n, p, adjoint=adjoint), power)
    assert_source_model(model, source)
    assert [(reg.name, reg.bitsize) for reg in source.signature.lefts()] == [
        ("w", 2),
        ("w_mean", 2),
        ("r", n),
        ("s", n),
    ]
    assert source.bloq_counts() == {Toffoli(): 15 if adjoint else 31, power: 2}
    assert [call.repetitions for call in model.calls] == [15 if adjoint else 31, 2]
    for operation in (power, source):
        with pytest.raises(DecomposeNotImplementedError):
            operation.decompose_bloq()


def test_projectile_kinetic_roundtrip_and_width_validation() -> None:
    forward = estimate_ftqc_resources(source_projectile_prepare_t_cost(6, 8))
    reverse = estimate_ftqc_resources(
        source_projectile_prepare_t_cost(6, 8, adjoint=True)
    )
    assert forward.logical.gates.toffoli == 47
    assert reverse.logical.gates.toffoli == 27
    assert (
        forward.logical.peak_logical_qubits == reverse.logical.peak_logical_qubits == 20
    )
    assert forward.emitted is reverse.emitted is None
    with pytest.raises(ValueError, match="at least the electron width"):
        source_projectile_prepare_t_cost(8, 6)


@pytest.mark.parametrize("n", [1, 2, 8])
@pytest.mark.ftqc_reference
def test_projectile_select_t_cost(n: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_t import (
        SelectTFirstQuantizationWithProj,
    )

    source = SelectTFirstQuantizationWithProj(n, 10)
    assert_source_model(source_projectile_select_t_cost(n), source)
    assert source.signature.get_left("w_mean").bitsize == 3
    assert source.signature.get_left("p").shape == (3,)
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()


@pytest.mark.parametrize(
    ("p", "n", "atoms", "precision"),
    [(2, 2, 1, 1), (2, 4, 5, 4), (3, 6, 10, 8), (6, 8, 10, 16), (4, 2, 3, 5)],
)
@pytest.mark.ftqc_reference
def test_projectile_select_uv_cost_graph(
    p: int, n: int, atoms: int, precision: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_uv import (
        SelectUVFirstQuantizationWithProj,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = SelectUVFirstQuantizationWithProj(p, n, 10, atoms, precision)
    model = source_projectile_select_uv_cost(p, n, atoms, precision)
    assert_source_model(model, source)
    assert model.qubits == source.signature.n_qubits()
    for name, width in (("p", n), ("q", p), ("nu", n + 1)):
        reg = source.signature.get_left(name)
        assert reg.bitsize == width and reg.shape == (3,)
    expected = Counter()
    for child, multiplicity in source.bloq_counts().items():
        counts = get_cost_value(child, QECGatesCost()).asdict()
        expected[
            (
                type(child).__name__,
                get_cost_value(child, QubitCount()),
                tuple(sorted(counts.items())),
            )
        ] += multiplicity
    actual = Counter()
    for (_, child), multiplicity in modeled_call_graph(
        model, max_depth=1
    ).direct_calls.items():
        counts = tuple(
            sorted((key, value) for key, value in asdict(child.gates).items() if value)
        )
        actual[(child.name, child.qubits, counts)] += multiplicity
    assert actual == expected
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()


@pytest.mark.parametrize(("p", "n"), [(1, 2), (2, 1)])
@pytest.mark.ftqc_reference
def test_projectile_select_uv_one_bit_source_failure(p: int, n: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_uv import (
        SelectUVFirstQuantizationWithProj,
    )
    from qualtran.resource_counting import QubitCount, get_cost_value

    with pytest.raises(RuntimeError, match="bitsize must be > 1"):
        get_cost_value(SelectUVFirstQuantizationWithProj(p, n, 10, 1, 8), QubitCount())
    with pytest.raises(ValueError, match="at least two momentum bits"):
        source_projectile_select_uv_cost(p, n, 1, 8)


@pytest.mark.parametrize(
    ("p", "n", "m"), [(1, 1, 3), (2, 2, 5), (2, 4, 8), (3, 5, 9), (6, 8, 524288)]
)
@pytest.mark.ftqc_reference
def test_projectile_prepare_nu_forward_and_adjoint_boundary(
    p: int, n: int, m: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import BloqError, DecomposeNotImplementedError, LeftDangle, Side
    from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_nu import (
        PrepareMuUnaryEncodedOneHotWithProj,
        PrepareNuStateWithProj,
    )
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    mu = PrepareMuUnaryEncodedOneHotWithProj(n, p)
    for adjoint in (False, True):
        leaf = mu.adjoint() if adjoint else mu
        assert leaf.signature.get_right("flag").side == Side.RIGHT
        assert_source_model(
            source_projectile_prepare_mu_cost(p, n, adjoint=adjoint), leaf
        )
        with pytest.raises(DecomposeNotImplementedError):
            leaf.decompose_bloq()

    source = PrepareNuStateWithProj(p, n, m)
    assert_source_model(source_projectile_prepare_nu_cost(p, n, m), source)
    graph = source.decompose_bloq()
    composite = source_projectile_prepare_nu_cost(p, n, m, decomposed=True)
    assert_source_model(composite, graph)
    live = {edge for edge in graph.connections if edge.left.binst == LeftDangle}
    stages = {}
    for node, incoming, outgoing in graph.iter_bloqnections():
        live.difference_update(incoming)
        name = str(node.bloq)
        if isinstance(node.bloq, PrepareUniformSuperposition):
            assert incoming[0].left.reg.name == outgoing[0].right.reg.name == "m"
            name = f"PrepareUniformSuperposition({m})"
        if name == "TestNuInequality":
            assert {
                (type(edge.left.binst.bloq).__name__, edge.right.reg.name)
                for edge in incoming
                if edge.right.reg.name.startswith("flag_")
            } == {
                ("PrepareMuUnaryEncodedOneHotWithProj", "flag_mu_prep"),
                ("FlagZeroAsFailure", "flag_minus_zero"),
                ("TestNuLessThanMu", "flag_ineq"),
            }
        stages[name] = sum(edge.shape for edge in live)
        live.update(outgoing)
    assert stages == {
        call.component.name: call.bystander_qubits for call in composite.calls
    }

    reverse = source.adjoint()
    with pytest.raises(BloqError, match="does not accept Soquets.*flag"):
        reverse.decompose_bloq()
    with pytest.raises(RuntimeError, match="does not accept Soquets.*flag"):
        get_cost_value(reverse, QubitCount())
    gates = source_projectile_prepare_nu_adjoint_gate_cost(p, n)
    assert {
        key: value for key, value in asdict(gates).items() if value
    } == get_cost_value(reverse, QECGatesCost()).asdict()


def test_projectile_prepare_nu_rejects_unbound_dimensions() -> None:
    with pytest.raises(ValueError, match="at least the electron width"):
        source_projectile_prepare_nu_cost(4, 2, 8)
    with pytest.raises(ValueError, match="m_param >= 2"):
        source_projectile_prepare_nu_cost(2, 4, 1)
