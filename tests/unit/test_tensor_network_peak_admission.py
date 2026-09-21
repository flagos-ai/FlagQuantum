"""A tensor-network run is bounded by the capacity it was admitted against.

The Runtime estimate for tensor-network residency is a floor and not a bound (see
``test_tensor_network_memory_estimate``), so what makes a run safe is capacity
rather than the estimate: a run that declares ``memory_limit_bytes`` executes
against that number as a hard peak budget, and a budget that cannot be met — or
that can be met only by slicing into a number of terms which dwarfs the memory it
saves — raises instead of running over it.

Three things are pinned here. First, the peak is a property of the contraction
order, so the shape-only measurement
(``tensor_network_contraction_peak_bytes``) reproduces the numerical build's peak
per order and allocates nothing. Second, the declared limit reaches the
contraction that actually runs, including the bra-operator-ket contractions behind
analytic measurement, which are a separate contraction from the state one and were
previously contracted with no budget at all. Third, meeting the limit by slicing
does not change the number that comes back.
"""

from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.runtime.options import ExecutionOptions
from flagquantum.simulation.tensor_network import (
    build_shape_only_tensor_network,
    run_tensor_network,
    tensor_network_contraction_peak_bytes,
)
from flagquantum.simulation.tensor_network.contraction import (
    contract_nodes_with_byte_budget,
)
from flagquantum.simulation.tensor_network.entrypoints import (
    tensor_network_expectation_ps,
)
from flagquantum.simulation.tensor_network.local import build_local_tensor_network

pytestmark = pytest.mark.unit

# 24 two-qubit gates on 8 wires: interaction width 3, so the Runtime proxy reports
# 512 bytes while the orders below measure 2048 to 131072 bytes.
_BONDED_EDGES: tuple[tuple[int, int], ...] = (
    (2, 4),
    (2, 1),
    (4, 2),
    (4, 0),
    (7, 3),
    (6, 2),
    (4, 5),
    (7, 4),
    (4, 6),
    (7, 0),
    (3, 5),
    (0, 7),
    (3, 7),
    (2, 4),
    (7, 1),
    (3, 6),
    (3, 0),
    (7, 0),
    (4, 5),
    (2, 1),
    (7, 6),
    (1, 4),
    (0, 7),
    (5, 4),
)

# The local runner's default order. A change here is a change to what the runner
# executes, so this number has to be re-measured rather than adjusted.
_RUNNER_DEFAULT_PEAK_BYTES = 131072

# The output of this program is 256 amplitudes, so no contraction of it can hold
# less than 2048 bytes and this is the floor the state path bottoms out on.
_STATE_FLOOR_BYTES = 2048

# A limit the state path can meet, above its own full materialization.
_WORKING_LIMIT_BYTES = 2048


def _bonded_circuit() -> fq.Circuit:
    circuit = fq.Circuit(8)
    circuit.h(0)
    for left, right in _BONDED_EDGES:
        circuit.cx(left, right)
    return circuit


def _rotated_circuit() -> fq.Circuit:
    """The same connectivity under single-qubit rotations, so no value is 0 or 1."""

    circuit = fq.Circuit(8)
    for wire, angle in enumerate((0.3, 1.1, 2.2, 0.7, 1.9, 2.9, 0.9, 1.4)):
        circuit.rx(wire, angle)
    for left, right in _BONDED_EDGES:
        circuit.cx(left, right)
    circuit.rz(0, 0.6)
    circuit.ry(3, 1.25)
    return circuit


def _measured_peak_bytes(circuit: Any, **kwargs: Any) -> int:
    plan = build_local_tensor_network(circuit)
    profile = plan.contraction_profile(**kwargs)
    element_size = max(int(node.tensor.element_size()) for node in plan.nodes)
    return int(profile.peak_size) * element_size


def _run_probabilities(circuit: fq.Circuit, limit: int | None) -> Any:
    options = (
        ExecutionOptions(mode="tensor_network")
        if limit is None
        else ExecutionOptions(mode="tensor_network", memory_limit_bytes=limit)
    )
    return fq.run(circuit, options=options, outputs=fq.probabilities())


def test_shape_only_measurement_reproduces_the_numerical_peak() -> None:
    circuit = _bonded_circuit()

    for strategy in ("greedy", "memory_greedy", "quality_multistart"):
        assert tensor_network_contraction_peak_bytes(
            circuit, contraction_strategy=strategy
        ) == _measured_peak_bytes(circuit, strategy=strategy), strategy


def test_the_default_order_is_the_runners_own() -> None:
    circuit = _bonded_circuit()

    assert (
        tensor_network_contraction_peak_bytes(circuit)
        == _RUNNER_DEFAULT_PEAK_BYTES
        == tensor_network_contraction_peak_bytes(circuit, contraction_strategy="greedy")
    )


def test_shape_only_measurement_compiles_nothing_into_the_program() -> None:
    circuit = _bonded_circuit()
    circuit.to_ir()
    before = dict(getattr(circuit, "_backend_programs", {}))

    build_shape_only_tensor_network(circuit)
    assert dict(getattr(circuit, "_backend_programs", {})) == before

    # The numerical build is the one that populates the backend program cache.
    build_local_tensor_network(circuit)
    assert [name for name, _ in getattr(circuit, "_backend_programs", {})] == [
        "tensor_network"
    ]


def test_the_declared_limit_binds_the_order_that_runs() -> None:
    circuit = _bonded_circuit()

    assert tensor_network_contraction_peak_bytes(circuit) > _WORKING_LIMIT_BYTES

    result = _run_probabilities(circuit, _WORKING_LIMIT_BYTES)

    assert result.probabilities.shape == (1, 1 << circuit.n_wires)
    # The order the executor ran fits the capacity, which the default order does not.
    assert (
        tensor_network_contraction_peak_bytes(
            circuit,
            contraction_strategy="quality_sliced",
            max_intermediate_bytes=_WORKING_LIMIT_BYTES,
        )
        == _WORKING_LIMIT_BYTES
    )


def test_a_limit_below_what_slicing_can_reach_raises() -> None:
    circuit = _bonded_circuit()

    # The program's own output is 256 amplitudes, so no slicing of the state
    # network can hold less than 2048 bytes. A smaller limit must refuse rather
    # than contract 131072 bytes under it.
    for limit in (_STATE_FLOOR_BYTES // 2, _STATE_FLOOR_BYTES // 4):
        with pytest.raises(ValueError, match="cannot satisfy effective peak budget"):
            run_tensor_network(circuit, max_intermediate_bytes=limit).state()


def test_the_limit_reaches_the_contraction_behind_analytic_measurement() -> None:
    circuit = _bonded_circuit()

    # Analytic probabilities are read off the bra-operator-ket network through one
    # marginal per subset of wires, not off the state. That is a separate
    # contraction and has to be held to the declared limit too; when it was not,
    # this run silently contracted 524288 bytes under a 256-byte admission.
    with pytest.raises(ValueError, match="rejected by economics preflight"):
        _run_probabilities(circuit, 256)


def test_an_undeclared_limit_enforces_nothing() -> None:
    circuit = _bonded_circuit()

    result = _run_probabilities(circuit, None)

    assert result.probabilities.shape == (1, 1 << circuit.n_wires)
    # No capacity was declared, so the run keeps the default order and its peak.
    assert tensor_network_contraction_peak_bytes(circuit) == _RUNNER_DEFAULT_PEAK_BYTES


def test_a_budgeted_contraction_reports_a_plan_inside_its_limit() -> None:
    plan = build_local_tensor_network(_bonded_circuit())

    _, slicing = contract_nodes_with_byte_budget(
        plan.nodes,
        plan.output_labels,
        max_peak_bytes=_WORKING_LIMIT_BYTES,
        contraction_strategy="greedy",
    )

    assert slicing.budget_satisfied
    assert slicing.peak_bytes <= _WORKING_LIMIT_BYTES
    assert slicing.target_peak_bytes == _WORKING_LIMIT_BYTES


def test_slicing_to_fit_the_limit_preserves_the_expectation() -> None:
    plan = build_local_tensor_network(_rotated_circuit())

    for wires in ((0,), (0, 1), (2, 5), tuple(range(8))):
        unbounded = tensor_network_expectation_ps(plan, z=wires)
        for limit in (_WORKING_LIMIT_BYTES, _WORKING_LIMIT_BYTES // 2):
            sliced = tensor_network_expectation_ps(plan, z=wires, max_peak_bytes=limit)
            assert sliced.shape == unbounded.shape
            assert torch.allclose(sliced, unbounded, atol=1e-6), (wires, limit)
