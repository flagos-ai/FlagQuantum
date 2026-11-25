"""Retain complete routing and traffic programs with independent factored costs."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.logistics.vehicle_routing_problem import (
    main as routing,
)
from examples.synthesis.applications.logistics.vehicle_routing_problem.comparisons import (
    circuit as routing_reference,
)
from examples.synthesis.applications.telecom.network_traffic_optimization import (
    main as traffic,
)
from examples.synthesis.applications.telecom.network_traffic_optimization.comparisons import (
    circuit as traffic_reference,
)
from examples.synthesis.applications.telecom.resiliency_planning import (
    main as resiliency,
)
from examples.synthesis.applications.telecom.resiliency_planning.comparisons import (
    circuit as resiliency_reference,
)
from examples.synthesis.applications.telecom.resiliency_planning.resiliency_planning_amd import (
    main as amd,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.factor_reference import (
    CostFactor,
    factored_cost_circuit,
    squared_residual,
)
from flagquantum import synth
from flagquantum.compiler import estimate_resources
from flagquantum.errors import PlanningError

pytestmark = pytest.mark.integration
CORPUS = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/corpus/original"


@dataclass(frozen=True)
class Case:
    path: str
    build: Callable[[tuple[float, ...]], synth.Program]
    reference: Callable[[tuple[float, ...]], fq.Circuit]
    bindings: tuple[tuple[float, ...], ...]
    register: str
    width: int


CASES = (
    Case(
        "applications/logistics/vehicle_routing_problem/vehicle_routing_problem.qmod",
        routing.build_program,
        routing_reference.reference_circuit,
        routing_reference.BINDINGS,
        "x",
        16,
    ),
    Case(
        "applications/telecom/network_traffic_optimization/network_traffic_optimization.qmod",
        traffic.build_program,
        traffic_reference.reference_circuit,
        traffic_reference.BINDINGS,
        "z",
        24,
    ),
    Case(
        "applications/telecom/resiliency_planning/resiliency_planning.qmod",
        resiliency.build_program,
        resiliency_reference.reference_circuit,
        resiliency_reference.BINDINGS,
        "z",
        24,
    ),
    Case(
        "applications/telecom/resiliency_planning/resiliency_planning_AMD.qmod",
        amd.build_program,
        resiliency_reference.reference_circuit,
        resiliency_reference.BINDINGS,
        "z",
        24,
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: Path(c.path).stem)
@pytest.mark.parametrize("dataset", [0, 1])
@pytest.mark.slow
def test_complete_original_and_native_costs_match_absolute_full_states(
    case: Case, dataset: int
) -> None:
    params = case.bindings[dataset]
    native = case.build(params)
    source = synth.synthesize(
        CORPUS / case.path,
        classical_inputs={"params": params},
        max_workspace=0,
        memory_limit_bytes=2**30,
    )
    expected = case.reference(params)
    for program in (native, source):
        compare_states(program, expected, (case.register,))
        assert program.report["workspace_qubits"] == 0
        assert estimate_resources(program.circuit).n_qubits == case.width


def test_factor_reference_matches_dense_four_bit_cost_with_absolute_constant() -> None:
    factors: tuple[CostFactor, ...] = (
        squared_residual((3, 1, 0), (-2.0, 1.0, 3.0), 1.0, 0.125),
        ((2, 0, 1), tuple(0.2 * int(i.bit_count() >= 2) for i in range(8))),
        ((), (0.37,)),
    )
    params = (-0.29, 0.41, 0.23, -0.17)
    values = []
    for basis in range(16):
        bits = tuple((basis >> i) & 1 for i in range(4))
        values.append(
            sum(
                table[sum(bits[bit] << i for i, bit in enumerate(support))]
                for support, table in factors
            )
        )
    dense = fq.Circuit(4, dtype=torch.complex128)
    for q in range(4):
        dense.h(q)
    for layer in range(2):
        dense.any(
            *range(4),
            matrix=torch.diag(
                torch.exp(
                    1j * params[2 * layer] * torch.tensor(values, dtype=torch.float64)
                )
            ),
        )
        for q in range(4):
            dense.rx(q, theta=0.1 * params[2 * layer + 1])
    reference = factored_cost_circuit(4, factors, params, mixer_scale=0.1)
    torch.testing.assert_close(
        fq.run(reference).statevector(), fq.run(dense).statevector(), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: Path(c.path).stem)
@pytest.mark.parametrize("dataset", [0, 1])
def test_both_objectives_retain_the_same_complete_circuit(
    case: Case, dataset: int
) -> None:
    params = case.bindings[dataset]
    inputs: dict[str, float | Sequence[float]] = {"params": params}
    if case.build is routing.build_program:
        entry = routing.main
        inputs.update(
            departure=routing.DEPARTURE,
            arrival=routing.ARRIVAL,
            distances=routing.DISTANCES,
        )
    elif case.build is traffic.build_program:
        entry = traffic.main
    else:
        entry = resiliency.main
    native = synth.synthesize(
        entry,
        classical_inputs=inputs,
        objective="depth",
        max_workspace=0,
        memory_limit_bytes=2**30,
    )
    assert native.circuit.instructions == case.build(params).circuit.instructions
    width = synth.synthesize(
        CORPUS / case.path,
        classical_inputs={"params": params},
        objective="width",
        memory_limit_bytes=2**30,
    )
    depth = synth.synthesize(
        CORPUS / case.path,
        classical_inputs={"params": params},
        objective="depth",
        memory_limit_bytes=2**30,
    )
    assert width.circuit.instructions == depth.circuit.instructions


def test_amd_named_source_reuses_the_identical_hardware_neutral_program() -> None:
    base = CORPUS / "applications/telecom/resiliency_planning"
    assert (base / "resiliency_planning.qmod").read_bytes() == (
        base / "resiliency_planning_AMD.qmod"
    ).read_bytes()
    assert amd.main is resiliency.main
    assert amd.build_program is resiliency.build_program


def test_congestion_is_a_cubic_capacity_threshold_including_all_three_paths() -> None:
    congestion = traffic_reference.cost_factors()[-8:]
    assert all(
        table == (0.0, 0.0, 0.0, 0.2, 0.0, 0.2, 0.2, 0.2) for _, table in congestion
    )
    assert congestion[0][0] == (0, 8, 16)
    assert congestion[-1][0] == (7, 15, 23)


@pytest.mark.parametrize(
    "max_qubits,max_operations,max_depth,memory_limit_bytes",
    [
        (23, None, None, None),
        (None, 100, None, None),
        (None, None, 20, None),
        (None, None, None, 1024),
    ],
)
def test_complete_traffic_program_preserves_actual_resource_budgets(
    max_qubits: int | None,
    max_operations: int | None,
    max_depth: int | None,
    memory_limit_bytes: int | None,
) -> None:
    with pytest.raises(
        (ValueError, PlanningError),
        match="budget|qubit|operation|depth|memory|limit|candidate",
    ):
        synth.synthesize(
            traffic.main,
            classical_inputs={"params": traffic_reference.BINDINGS[0]},
            max_qubits=max_qubits,
            max_operations=max_operations,
            max_depth=max_depth,
            memory_limit_bytes=memory_limit_bytes,
        )
