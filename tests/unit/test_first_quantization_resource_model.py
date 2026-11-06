"""Check first-quantized SELECT and PREPARE models against the original cost graph."""

from dataclasses import asdict

import pytest

from flagquantum.algorithms.first_quantization_resource_model import (
    source_apply_nuclear_phase_cost,
    source_prepare_ij_cost,
    source_prepare_power_two_cost,
    source_prepare_t_cost,
    source_prepare_tuv_cost,
    source_prepare_uv_cost,
    source_prepare_zeta_cost,
    source_select_first_quantization_cost,
    source_select_t_cost,
    source_select_uv_cost,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

pytestmark = pytest.mark.unit


def assert_source_model(component: ComponentCost, source: object) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    report = estimate_ftqc_resources(component)
    counts = {k: v for k, v in asdict(report.logical.gates).items() if v}
    assert counts == get_cost_value(source, QECGatesCost()).asdict()
    assert report.logical.peak_logical_qubits == get_cost_value(source, QubitCount())
    assert report.emitted is None


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("p", [1, 2, 6])
def test_kinetic_source_cost(p: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.chemistry.pbc.first_quantization.select_t import (
        SelectTFirstQuantization,
    )

    source = SelectTFirstQuantization(p, 3)
    assert_source_model(source_select_t_cost(p), source)
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(("p", "nuc"), [(1, 2), (2, 1), (4, 4), (4, 8)])
def test_nuclear_phase_source_cost(p: int, nuc: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.pbc.first_quantization.select_uv import (
        ApplyNuclearPhase,
    )

    assert_source_model(
        source_apply_nuclear_phase_cost(p, nuc), ApplyNuclearPhase(p, nuc)
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("p", "atoms", "nuc"), [(2, 1, 1), (2, 2, 16), (4, 5, 4), (2, 2, 100)]
)
def test_potential_source_cost(p: int, atoms: int, nuc: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.chemistry.pbc.first_quantization.select_uv import (
        SelectUVFirstQuantization,
    )

    source = SelectUVFirstQuantization(p, 3, atoms, nuc)
    assert_source_model(source_select_uv_cost(p, atoms, nuc), source)
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("p", "eta", "atoms", "m", "nuc"),
    [
        (2, 3, 2, 256, 16),
        (2, 5, 1, 1, 1),
        (3, 9, 3, 17, 1),
        (6, 10, 10, 256, 16),
        (2, 2, 2, 1, 100),
    ],
)
def test_select_graph_and_peak_width(
    p: int, eta: int, atoms: int, m: int, nuc: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import LeftDangle
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare import (
        SelectFirstQuantization,
    )

    source = SelectFirstQuantization(p, eta, atoms, 1, m_param=m, num_bits_nuc_pos=nuc)
    model = source_select_first_quantization_cost(
        p, eta, atoms, m_param=m, num_bits_nuc_pos=nuc
    )
    assert_source_model(model, source)
    assert model.qubits == source.signature.n_qubits()

    graph = source.decompose_bloq()
    live = {edge for edge in graph.connections if edge.left.binst == LeftDangle}
    stages = []
    for node, incoming, outgoing in graph.iter_bloqnections():
        live.difference_update(incoming)
        if not isinstance(node.bloq, (Allocate, Free)):
            stages.append((str(node.bloq), sum(edge.shape for edge in live)))
        live.update(outgoing)
    assert stages == [
        (call.component.name, call.bystander_qubits) for call in model.calls
    ]


@pytest.mark.ftqc_reference
def test_source_one_bit_momentum_width_failure_is_explicit() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare import (
        SelectFirstQuantization,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = SelectFirstQuantization(1, 2, 2, 2)
    assert get_cost_value(source, QECGatesCost()).toffoli == 101
    with pytest.raises(RuntimeError, match="bitsize must be > 1"):
        get_cost_value(source, QubitCount())
    with pytest.raises(ValueError, match="at least two momentum bits"):
        source_select_first_quantization_cost(1, 2, 2)


@pytest.mark.parametrize(("p", "eta", "precision"), [(1, 2, 1), (2, 3, 5), (6, 10, 8)])
@pytest.mark.ftqc_reference
def test_prepare_t_cost_paths(p: int, eta: int, precision: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError, LeftDangle, RightDangle
    from qualtran.bloqs.basic_gates import Toffoli
    from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_t import (
        PreparePowerTwoState,
        PrepareTFirstQuantization,
    )
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )

    power = PreparePowerTwoState(p)
    assert_source_model(source_prepare_power_two_cost(p), power)
    with pytest.raises(DecomposeNotImplementedError):
        power.decompose_bloq()

    source = PrepareTFirstQuantization(p, eta, precision)
    declared = source_prepare_t_cost(p)
    assert_source_model(declared, source)
    assert source.bloq_counts() == {Toffoli(): 13, power: 2}
    assert [(call.component.name, call.repetitions) for call in declared.calls] == [
        ("Toffoli", 13),
        ("PreparePowerTwoState", 2),
    ]

    composite = source.decompose_bloq()
    assert_source_model(source_prepare_t_cost(p, decomposed=True), composite)
    # The three operations are disjoint. Check register bindings independently
    # of their ordering in the composite's topological traversal.
    assert {
        (node.bloq, incoming[0].left.reg.name, outgoing[0].right.reg.name)
        for node, incoming, outgoing in composite.iter_bloqnections()
        if incoming[0].left.binst == LeftDangle
        and outgoing[0].right.binst == RightDangle
    } == {
        (PrepareUniformSuperposition(3), "w", "w"),
        (power, "r", "r"),
        (power, "s", "s"),
    }


@pytest.mark.parametrize(("eta", "precision"), [(1, 8), (2, 1), (10, 8)])
@pytest.mark.ftqc_reference
def test_prepare_ij_cost(eta: int, precision: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.chemistry.pbc.first_quantization import (
        prepare,
        select_and_prepare,
    )

    for source_module in (prepare, select_and_prepare):
        source = source_module.UniformSuperpostionIJFirstQuantization(eta, precision)
        assert_source_model(source_prepare_ij_cost(eta, precision), source)
        with pytest.raises(DecomposeNotImplementedError):
            source.decompose_bloq()


def test_kinetic_prepare_resource_report_separates_emitted_coordinate() -> None:
    from flagquantum.compiler.registers.uniform_preparation import (
        uniform_preparation_ir,
        uniform_preparation_requirements,
    )
    from flagquantum.core.ir import CircuitIR

    target, work, slots = uniform_preparation_requirements(3)
    coordinate = uniform_preparation_ir(
        3,
        tuple(range(target)),
        tuple(range(target, target + work)),
        tuple(range(slots)),
    )
    declared = estimate_ftqc_resources(source_prepare_t_cost(6))
    composite = estimate_ftqc_resources(source_prepare_t_cost(6, decomposed=True))
    emitted = estimate_ftqc_resources(
        source_prepare_t_cost(6, decomposed=True).calls[0].component,
        emitted_ir=CircuitIR(target + work, coordinate),
    )
    assert declared.logical.gates.toffoli == 21
    assert composite.logical.gates.toffoli == 8
    assert (
        declared.logical.peak_logical_qubits
        == composite.logical.peak_logical_qubits
        == 18
    )
    assert declared.emitted is composite.emitted is None
    assert emitted.emitted is not None
    assert emitted.emitted.t_count == 20
    assert emitted.emitted.unconditional["measure"] == 5
    assert emitted.emitted.declared_qubits == 6
    ij = estimate_ftqc_resources(source_prepare_ij_cost(10))
    assert 2 * (ij.logical.gates.toffoli + declared.logical.gates.toffoli) == 126


@pytest.mark.parametrize(
    ("atoms", "charge", "precision"),
    [(1, 0, 1), (1, 1, 8), (2, 4, 8), (9, 10, 16), (33, 15, 2), (2, 2**54 + 1, 8)],
)
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.ftqc_reference
def test_prepare_zeta_cost(
    atoms: int, charge: int, precision: int, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError, Side
    from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_zeta import (
        PrepareZetaState,
    )

    source = PrepareZetaState(atoms, charge, precision)
    if adjoint:
        source = source.adjoint()
    index = source.signature.get_left("l")
    assert index.side == Side.THRU
    assert index.bitsize == (atoms - 1).bit_length()
    assert_source_model(
        source_prepare_zeta_cost(atoms, charge, adjoint=adjoint), source
    )
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()


@pytest.mark.parametrize(
    ("p", "eta", "atoms", "m", "charge", "precision"),
    [
        (1, 2, 1, 3, 0, 1),
        (2, 4, 1, 2, 1, 16),
        (2, 3, 2, 5, 4, 8),
        (3, 4, 33, 9, 15, 2),
        (6, 10, 10, 32768, 10, 8),
    ],
)
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.ftqc_reference
def test_prepare_uv_source_graph(
    p: int, eta: int, atoms: int, m: int, charge: int, precision: int, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import LeftDangle, RightDangle
    from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_uv import (
        PrepareUVFirstQuantization,
    )

    source = PrepareUVFirstQuantization(p, eta, atoms, m, charge, precision)
    if adjoint:
        source = source.adjoint()
    model = source_prepare_uv_cost(p, atoms, m, charge, adjoint=adjoint)
    assert_source_model(model, source)
    assert model.qubits == source.signature.n_qubits()
    composite = source.decompose_bloq()
    assert_source_model(model, composite)
    live = {edge for edge in composite.connections if edge.left.binst == LeftDangle}
    stages = []
    for node, incoming, outgoing in composite.iter_bloqnections():
        live.difference_update(incoming)
        assert all(edge.left.binst == LeftDangle for edge in incoming)
        assert all(edge.right.binst == RightDangle for edge in outgoing)
        register_names = {edge.left.reg.name for edge in incoming}
        assert register_names == {edge.right.reg.name for edge in outgoing}
        assert register_names == (
            {"l"}
            if str(node.bloq) == "PrepareZetaState"
            else {"mu", "nu", "m", "flag_nu"}
        )
        stages.append((str(node.bloq), sum(edge.shape for edge in live)))
        live.update(outgoing)
    assert stages == [
        (call.component.name, call.bystander_qubits) for call in model.calls
    ]


def test_prepare_uv_roundtrip_resources() -> None:
    forward = estimate_ftqc_resources(source_prepare_uv_cost(6, 10, 32768, 10))
    reverse = estimate_ftqc_resources(
        source_prepare_uv_cost(6, 10, 32768, 10, adjoint=True)
    )
    assert (
        forward.logical.gates.total_t_count() + reverse.logical.gates.total_t_count()
        == 2516
    )
    assert forward.logical.peak_logical_qubits == 77
    assert reverse.logical.peak_logical_qubits == 50
    assert forward.emitted is reverse.emitted is None


@pytest.mark.parametrize(
    ("bits", "eta", "charge", "precision"),
    [(1, 1, 0, 1), (8, 3, 0, 5), (16, 10, 10, 8), (16, 4, 2**54, 8)],
)
@pytest.mark.ftqc_reference
def test_prepare_tuv_source_cost(
    bits: int, eta: int, charge: int, precision: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare import (
        PrepareTUVSuperpositions,
    )

    source = PrepareTUVSuperpositions(bits, eta, charge, precision)
    model = source_prepare_tuv_cost(bits, eta, charge, precision)
    assert [(reg.name, reg.bitsize) for reg in source.signature.lefts()] == [
        ("tuv", 1),
        ("uv", 1),
    ]
    for operation in (source, source.adjoint()):
        assert_source_model(model, operation)
        with pytest.raises(DecomposeNotImplementedError):
            operation.decompose_bloq()
