"""Check first-quantized SELECT models against the original cost graph."""

from dataclasses import asdict

import pytest

from flagquantum.algorithms.first_quantization_resource_model import (
    source_apply_nuclear_phase_cost,
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
