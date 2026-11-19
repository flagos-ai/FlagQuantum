"""Identical candidate sources reuse checked target work within one request."""

from collections.abc import Callable
from dataclasses import replace

import pytest

from flagquantum.compiler import estimate_resources
from flagquantum.compiler._construction.codegen.arithmetic import Schedule
from flagquantum.compiler._construction.codegen.registers import RegisterCandidate
from flagquantum.compiler._construction.scheduling import selection
from flagquantum.compiler._construction.scheduling.selection import (
    Constraints,
    ConstructionBudget,
    Selection,
)
from flagquantum.compiler._construction.scheduling.target import (
    Target,
    TargetError,
    TargetProduct,
    compile_target,
)
from flagquantum.compiler._construction.verification.constraints import CircuitLimits
from flagquantum.compiler.topology.coupling import CouplingMap
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def choose(
    build: Callable[[Schedule], RegisterCandidate],
    *,
    budget: ConstructionBudget | None = None,
    target: Target | None = None,
    constraints: Constraints | None = None,
) -> Selection:
    return selection._select_candidates(
        build,
        ("fixture",),
        {"q": (0, 1)},
        2,
        constraints if constraints is not None else Constraints(),
        budget if budget is not None else ConstructionBudget(),
        target if target is not None else Target(),
        "width",
    )


def candidate(strategy: Schedule) -> RegisterCandidate:
    return RegisterCandidate(
        CircuitIR(
            2,
            (
                Instruction("h", (0,)),
                Instruction("cx", (0, 1)),
                Instruction("rz", (1,), params={"theta": 0.37}),
            ),
        ),
        strategy,
        {"q": (0, 1)},
        (),
        0,
        0,
    )


@pytest.fixture
def compilations(monkeypatch: pytest.MonkeyPatch) -> list[TargetProduct]:
    products: list[TargetProduct] = []

    def record(source: CircuitIR, target: Target, limit: int) -> TargetProduct:
        product = compile_target(source, target, limit)
        products.append(product)
        return product

    monkeypatch.setattr(selection, "compile_target", record)
    return products


@pytest.mark.parametrize(
    "target",
    [
        Target(),
        Target(basis=("h", "u3", "cx", "rz", "phase")),
        Target(couplings=(CouplingMap(2, ((0, 1),)),)),
    ],
)
def test_equal_sources_preserve_both_checked_rows(
    compilations: list[TargetProduct], target: Target
) -> None:
    report = choose(candidate, target=target)
    assert report.complete and report.selected is report.rows[0]
    assert [row.strategy for row in report.rows] == ["retain", "recompute"]
    assert len(compilations) == 1
    for row in report.rows:
        assert row.arithmetic is not None and row.target is not None
        assert row.arithmetic.strategy == row.strategy
        row.target.validate()
        assert row.resources == estimate_resources(row.target.executable)
        assert row.stage_resources[-1] == ("executable", row.resources)
        assert not row.diagnostics


@pytest.mark.parametrize("change", ["angle", "width", "metadata", "preparation"])
def test_source_differences_are_not_merged(
    compilations: list[TargetProduct], change: str
) -> None:
    def build(strategy: Schedule) -> RegisterCandidate:
        row = candidate(strategy)
        if strategy == "retain":
            return row
        circuit = row.circuit
        if change == "angle":
            circuit = replace(
                circuit,
                instructions=(
                    *circuit.instructions[:-1],
                    Instruction("rz", (1,), params={"theta": 0.41}),
                ),
            )
        elif change == "width":
            return replace(row, peak_workspace=1)
        elif change == "metadata":
            circuit = replace(circuit, metadata={"global_phase": 0.23})
        else:
            circuit = replace(
                circuit, instructions=(Instruction("x", (1,)), *circuit.instructions)
            )
        return replace(row, circuit=circuit)

    report = choose(build)
    assert report.complete and report.selected is not None
    assert len(compilations) == 2
    assert report.rows[0].target is not report.rows[1].target


def test_later_requests_recompile_even_with_equal_source(
    compilations: list[TargetProduct],
) -> None:
    first = choose(candidate)
    second = choose(candidate)
    third = choose(candidate, target=Target(optimize=False))
    assert all(report.selected is not None for report in (first, second, third))
    assert len(compilations) == 3
    assert first.rows[0].target is not second.rows[0].target
    rejected = choose(candidate, budget=ConstructionBudget(operations=1))
    assert not rejected.complete and rejected.selected is None
    assert all(row.diagnostics[0].code == "target.budget" for row in rejected.rows)


def test_candidate_budget_and_each_workspace_constraint_still_apply(
    compilations: list[TargetProduct],
) -> None:
    report = choose(candidate, budget=ConstructionBudget(candidate_checks=1))
    assert len(compilations) == 1
    assert not report.complete and report.selected is None
    assert report.rows[1].diagnostics[0].code == "selection.budget"
    rejected = choose(
        candidate, constraints=Constraints(CircuitLimits(max_operations=1))
    )
    assert len(compilations) == 2
    assert rejected.complete and rejected.selected is None
    assert len(rejected.recoveries) == 2
    assert all(row.diagnostics for row in rejected.rows)


def test_cached_stage_tamper_invalidates_entire_search(
    compilations: list[TargetProduct],
) -> None:
    def build(strategy: Schedule) -> RegisterCandidate:
        if strategy == "recompute":
            metadata = compilations[0].submitted.metadata
            assert isinstance(metadata, dict)
            metadata["tampered"] = True
        return candidate(strategy)

    rejected = choose(build)
    assert len(compilations) == 1
    assert not rejected.complete and rejected.selected is None
    assert rejected.rows[1].diagnostics[0].code == "target.identity"


def test_post_selection_tamper_is_still_detected(
    compilations: list[TargetProduct],
) -> None:
    report = choose(candidate)
    metadata = compilations[0].executable.instructions[0].metadata
    assert isinstance(metadata, dict)
    metadata["tampered"] = True
    for row in report.rows:
        assert row.target is not None
        with pytest.raises(TargetError, match="changed"):
            row.target.validate()
