"""Three complete portfolio sources preserve signs, numeric fields and angles."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

import flagquantum as fq
from examples.synthesis.applications.finance.portfolio_optimization import main as app
from examples.synthesis.applications.finance.portfolio_optimization.comparisons import (
    circuit as app_ref,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.factor_reference import CostFactor
from examples.synthesis.tutorials.workshops.finance_workshops.combi_workshop_equality_constriants_po import (
    main as equality,
)
from examples.synthesis.tutorials.workshops.finance_workshops.combi_workshop_equality_constriants_po.comparisons import (
    circuit as equality_ref,
)
from examples.synthesis.tutorials.workshops.finance_workshops.combi_workshop_inequality_constriants_po import (
    main as inequality,
)
from examples.synthesis.tutorials.workshops.finance_workshops.combi_workshop_inequality_constriants_po.comparisons import (
    circuit as inequality_ref,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._construction.frontend_python import _QuantumFunction

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Case:
    name: str
    source: str
    definition: _QuantumFunction
    bindings: tuple[tuple[float, ...], ...]
    reference: Callable[[tuple[float, ...]], fq.Circuit]
    factors: Callable[[], list[CostFactor]]
    oracle: Callable[[tuple[int, ...]], float]
    width: int
    native_fields: tuple[str, ...]
    source_fields: tuple[str, ...]


CASES = (
    Case(
        "application",
        "applications/finance/portfolio_optimization/portfolio_optimization.qmod",
        app.main,
        app_ref.BINDINGS,
        app_ref.reference_circuit,
        app_ref.cost_factors,
        app_ref.classical_cost,
        12,
        ("a0", "a1", "a2", "slack"),
        ("v.w[0]", "v.w[1]", "v.w[2]", "v.budget_rule_slack_var"),
    ),
    Case(
        "inequality",
        "tutorials/workshops/finance_workshops/combi_workshop_Inequality_constriants_PO.qmod",
        inequality.main,
        inequality_ref.BINDINGS,
        inequality_ref.reference_circuit,
        inequality_ref.cost_factors,
        inequality_ref.classical_cost,
        9,
        ("a0", "a1", "a2", "slack"),
        ("w_array.a[0]", "w_array.a[1]", "w_array.a[2]", "w_array.slack"),
    ),
    Case(
        "equality",
        "tutorials/workshops/finance_workshops/combi_workshop_equality_constriants_PO.qmod",
        equality.main,
        equality_ref.BINDINGS,
        equality_ref.reference_circuit,
        equality_ref.cost_factors,
        equality_ref.classical_cost,
        6,
        ("a0", "a1", "a2"),
        ("w_array.a[0]", "w_array.a[1]", "w_array.a[2]"),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_every_classical_cost_input_preserves_covariance_returns_and_slack(
    case: Case,
) -> None:
    factors = case.factors()
    assert max(len(bits) for bits, _ in factors) <= 6
    for basis in range(1 << case.width):
        bits = tuple((basis >> i) & 1 for i in range(case.width))
        value = sum(
            table[sum(bits[bit] << i for i, bit in enumerate(support))]
            for support, table in factors
        )
        assert value == pytest.approx(case.oracle(bits), abs=1e-10, rel=0)


def test_originals_have_distinct_phase_penalties_and_slack_encodings() -> None:
    assert (
        app_ref.SIGN == -1
        and app_ref.PENALTY == 10
        and app_ref.SLACK_WEIGHTS == (1.0, 2.0, 3.0)
    )
    assert not app_ref.INTERLEAVED
    assert (
        inequality_ref.SIGN == 1
        and inequality_ref.PENALTY == 30.5
        and inequality_ref.SLACK_WEIGHTS == (1.0, 2.0, 4.0)
    )
    assert (
        equality_ref.SIGN == 1
        and equality_ref.PENALTY == 2
        and not equality_ref.SLACK_WEIGHTS
    )
    assert inequality_ref.INTERLEAVED and equality_ref.INTERLEAVED


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("dataset", [0, 1])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_complete_original_and_native_match_raw_independent_circuit(
    case: Case,
    dataset: int,
    objective: Literal["width", "depth"],
) -> None:
    params = case.bindings[dataset]
    native = synth.synthesize(
        case.definition,
        classical_inputs={"params": params},
        objective=objective,
        max_workspace=0,
    )
    source = synth.synthesize(
        ROOT / "tests/fixtures/classiq_qmod/corpus/original" / case.source,
        classical_inputs={"params": params},
        objective=objective,
        max_workspace=0,
    )
    reference = case.reference(params)
    compare_states(native, reference, case.native_fields)
    compare_states(source, reference, case.source_fields)
    for program in (native, source):
        resource = program.report["resources"]
        assert isinstance(resource, ResourceEstimate)
        assert resource.n_qubits == case.width
        assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_actual_qubit_operation_and_memory_budgets_reject(case: Case) -> None:
    params = case.bindings[0]
    with pytest.raises(ValueError):
        synth.synthesize(
            case.definition,
            classical_inputs={"params": params},
            max_qubits=case.width - 1,
        )
    with pytest.raises(ValueError):
        synth.synthesize(
            case.definition, classical_inputs={"params": params}, max_operations=1
        )
    with pytest.raises(ValueError):
        synth.synthesize(
            case.definition, classical_inputs={"params": params}, memory_limit_bytes=1
        )
