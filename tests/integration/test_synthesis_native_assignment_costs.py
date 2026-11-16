"""Readable native assignment costs retain original fields, layers and phases."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.logistics.traveling_salesman_problem import (
    main as routing,
)
from examples.synthesis.applications.logistics.traveling_salesman_problem.comparisons import (
    circuit as routing_ref,
)
from examples.synthesis.applications.optimization.kidney_exchange.kidney_exchange_problem import (
    main as kidney,
)
from examples.synthesis.applications.optimization.kidney_exchange.kidney_exchange_problem.comparisons import (
    circuit as kidney_ref,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.factor_reference import CostFactor
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._construction.frontend_python import _QuantumFunction

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Case:
    source: str
    entry: _QuantumFunction
    build: Callable[[tuple[float, ...]], synth.Program]
    reference: Callable[[tuple[float, ...]], fq.Circuit]
    source_reference: Callable[[tuple[float, ...]], fq.Circuit]
    cost: Callable[[tuple[int, ...]], float]
    factors: Callable[[], list[CostFactor]]
    bindings: tuple[tuple[float, ...], ...]
    source_fields: tuple[str, ...]
    register: str
    width: int


CASES = (
    Case(
        "applications/optimization/kidney_exchange/kidney_exchange_problem.qmod",
        kidney.main,
        kidney.build_program,
        kidney_ref.reference_circuit,
        kidney_ref.source_reference_circuit,
        kidney_ref.cost,
        kidney_ref.cost_factors,
        kidney_ref.BINDINGS,
        kidney_ref.SOURCE_FIELDS,
        "matches",
        9,
    ),
    Case(
        "applications/logistics/traveling_salesman_problem/traveling_salesman_problem.qmod",
        routing.main,
        routing.build_program,
        routing_ref.reference_circuit,
        routing_ref.source_reference_circuit,
        routing_ref.cost,
        routing_ref.cost_factors,
        routing_ref.BINDINGS,
        routing_ref.SOURCE_FIELDS,
        "visits",
        16,
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.register)
@pytest.mark.parametrize("binding", [0, 1])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_complete_original_and_native_against_independent_circuits(
    case: Case, binding: int, objective: Literal["width", "depth"]
) -> None:
    params = case.bindings[binding]
    native = synth.synthesize(
        case.entry,
        classical_inputs={"params": params},
        objective=objective,
        max_workspace=0,
    )
    source = synth.synthesize(
        ROOT / "tests/fixtures/classiq_qmod/corpus/original" / case.source,
        classical_inputs={"params": params},
        objective=objective,
    )
    compare_states(native, case.reference(params), (case.register,))
    compare_states(source, case.source_reference(params), case.source_fields)
    resources = native.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == case.width
    assert native.report["workspace_qubits"] == 0


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.register)
def test_every_classical_assignment_matches_independent_cost_factors(
    case: Case,
) -> None:
    factors = case.factors()
    for basis in range(1 << case.width):
        bits = tuple((basis >> bit) & 1 for bit in range(case.width))
        actual = sum(
            table[sum(bits[bit] << index for index, bit in enumerate(support))]
            for support, table in factors
        )
        assert actual == pytest.approx(case.cost(bits), abs=1e-12, rel=0)


def test_application_constraints_and_nonclosing_source_tour() -> None:
    assert kidney_ref.cost((1, 0, 0, 0, 1, 0, 0, 0, 1)) == pytest.approx(2.35)
    assert kidney_ref.cost((1, 1, 0, 0, 0, 0, 0, 0, 0)) == pytest.approx(-0.4)
    assert routing_ref.cost((0,) * 16) == -8
    assert (
        routing_ref.cost(tuple(int(bit in (0, 5, 10, 15)) for bit in range(16))) == -3
    )
    assert (
        routing_ref.cost(tuple(int(bit in (3, 6, 9, 12)) for bit in range(16))) == -11
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.register)
def test_zero_angles_remain_uniform_and_invalid_bindings_refuse(case: Case) -> None:
    program = case.build((0.0,) * len(case.bindings[0]))
    circuit = fq.Circuit(case.width, dtype=torch.complex128)
    for wire in range(case.width):
        circuit.h(wire)
    compare_states(program, circuit, (case.register,))
    with pytest.raises(ValueError, match="parameters"):
        case.build((0.1,))
    with pytest.raises(ValueError, match="finite"):
        case.build((float("nan"),) * len(case.bindings[0]))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.register)
@pytest.mark.parametrize(
    "budget", ["max_qubits", "max_operations", "max_depth", "memory_limit_bytes"]
)
def test_actual_program_budgets_refuse(case: Case, budget: str) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(
            case.entry,
            classical_inputs={"params": case.bindings[0]},
            max_qubits=case.width - 1 if budget == "max_qubits" else None,
            max_operations=1 if budget == "max_operations" else None,
            max_depth=1 if budget == "max_depth" else None,
            memory_limit_bytes=1 if budget == "memory_limit_bytes" else None,
        )
