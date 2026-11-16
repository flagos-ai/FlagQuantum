"""Check THC SELECT dimension-only estimates against the locked call graph."""

from dataclasses import asdict

import pytest

from flagquantum.algorithms.primitives.multiplexers.thc_resource_model import (
    select_thc_cost,
    select_thc_signature,
    thc_rotations_cost,
    thc_rotations_signature,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from tests.unit.test_sparse_chemistry_prepare import _assert_sparse_signature

pytestmark = pytest.mark.unit


def test_thc_model_without_tables() -> None:
    estimate = estimate_component_resources(select_thc_cost(2, 4, 8, 1))
    assert estimate.gates.toffoli == 107
    assert estimate.gates.cswap == 11
    assert estimate.gates.clifford == 1
    assert estimate.gates.measurement == 0
    assert estimate.gates.total_t_count() == 472
    assert estimate.peak_logical_qubits == 24


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "num_mu,spin_orb,bits,keep,kr1,kr2",
    [
        (2, 4, 8, 1, 1, 1),
        (10, 8, 8, 10, 4, 2),
        (6, 10, 12, 3, 3, 5),
        (1, 2, 8, 1, 2, 3),
    ],
)
@pytest.mark.parametrize("control_value", [None, 0, 1])
def test_thc_model_matches_locked_source(
    num_mu: int,
    spin_orb: int,
    bits: int,
    keep: int,
    kr1: int,
    kr2: int,
    control_value: int | None,
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import LeftDangle
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.bloqs.chemistry.thc.select_bloq import SelectTHC, THCRotations
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = SelectTHC(
        num_mu, spin_orb, bits, keep, kr1, kr2, control_val=control_value
    )
    signature = select_thc_signature(
        num_mu, spin_orb, keep, control_value=control_value
    )
    _assert_sparse_signature(signature, source.signature)
    graph = source.decompose_bloq()
    if control_value is not None:
        assert graph.final_soqs()["control"].binst == LeftDangle
    stages = tuple(graph.iter_bloqsoqs())
    rotations = [
        (ports, outputs)
        for node, ports, outputs in stages
        if isinstance(node.bloq, THCRotations)
    ]
    assert len(rotations) == 4
    assert isinstance(rotations[0][0]["data"].binst.bloq, Allocate)
    for (_, previous), (ports, _) in zip(rotations[:-1], rotations[1:], strict=True):
        assert ports["data"] == previous[1]
    (free_ports,) = (ports for node, ports, _ in stages if isinstance(node.bloq, Free))
    assert free_ports["reg"] == rotations[-1][1][1]
    component = select_thc_cost(
        num_mu, spin_orb, bits, keep, kr1=kr1, kr2=kr2, control_value=control_value
    )
    estimate = estimate_component_resources(component)
    assert {key: value for key, value in asdict(estimate.gates).items() if value} == (
        get_cost_value(source, QECGatesCost()).asdict()
    )
    assert estimate.peak_logical_qubits == get_cost_value(source, QubitCount())

    leaves = [
        call.component
        for call in component.calls
        if call.component.name.startswith("THCRotations(")
    ]
    assert len(leaves) == 4
    for leaf, (two_body, adjoint) in zip(
        leaves,
        ((False, False), (False, True), (True, False), (True, True)),
        strict=True,
    ):
        rotation = thc_rotations_cost(
            num_mu,
            spin_orb,
            bits,
            kr1=kr1,
            kr2=kr2,
            two_body_only=two_body,
            adjoint=adjoint,
        )
        assert leaf == rotation
        source_rotation = THCRotations(
            num_mu, spin_orb, bits, kr1, kr2, two_body, adjoint
        )
        _assert_sparse_signature(
            thc_rotations_signature(num_mu, spin_orb, bits), source_rotation.signature
        )
        costs = estimate_component_resources(rotation)
        source_gates = get_cost_value(source_rotation, QECGatesCost())
        assert {
            key: value for key, value in asdict(costs.gates).items() if value
        } == source_gates.asdict()
        assert costs.gates.total_t_count() == source_gates.total_t_count()
        assert costs.peak_logical_qubits == get_cost_value(
            source_rotation, QubitCount()
        )


def test_thc_rotations_can_be_estimated_independently() -> None:
    from flagquantum.qec.resources.estimation import estimate_ftqc_resources

    model = thc_rotations_cost(10, 8, 8, kr1=4, kr2=2, two_body_only=True, adjoint=True)
    report = estimate_ftqc_resources(model)
    assert report.logical.gates.toffoli == 56
    assert report.logical.gates.total_t_count() == 224
    assert report.logical.peak_logical_qubits == 17
    assert report.emitted is None
