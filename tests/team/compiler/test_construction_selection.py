"""Finite candidates, real target costs, deterministic policy and recovery."""

from collections.abc import Callable
from dataclasses import replace
from typing import Any, NoReturn

import pytest

from flagquantum.compiler import ResourceEstimate, estimate_resources
from flagquantum.compiler._construction.frontend.python import capture
from flagquantum.compiler._construction.scheduling.selection import (
    Constraints,
    ConstructionBudget,
    Selection,
    select_arithmetic,
)
from flagquantum.compiler._construction.scheduling.target import (
    Target,
    TargetProduct,
    compile_target,
)
from flagquantum.compiler._construction.verification.constraints import CircuitLimits
from flagquantum.compiler.basis_conversion import BasisConversionError
from flagquantum.compiler.topology.coupling import CouplingMap
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode
from tools.synthesis_compiler_examples.selection.main import (
    addition,
    polynomial,
    pressure,
    product,
)

pytestmark = pytest.mark.unit
BASIS = ("x", "h", "s", "sdg", "t", "tdg", "ry", "rz", "cx", "u3", "swap")


def branch_pressure(x: int, y: int) -> int:
    """Shared dependencies preserve a real schedule tradeoff after terminal XOR."""
    t = x * y
    a = t + x
    b = t + y
    return a * b


def _select(**kwargs: Any) -> Selection:
    """Exercise valid selection controls and intentionally invalid test inputs."""
    graph = capture(branch_pressure, {"x": 1, "y": 1})
    return select_arithmetic((graph,), entry=graph.name, **kwargs)


@pytest.mark.parametrize("function", [addition, product, polynomial, pressure])
@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("target", [Target(), Target(basis=BASIS)])
def test_complete_cost_matrix_and_nondominated_default(
    function: Callable[[int, int], int], width: int, target: Target
) -> None:
    graph = capture(function, {"x": width, "y": width + (function is addition)})
    result = select_arithmetic((graph,), entry=graph.name, target=target)
    assert result.complete and result.selected is not None
    assert [r.strategy for r in result.rows] == ["retain", "recompute"]
    for row in result.rows:
        assert row.feasible and row.target is not None and row.arithmetic is not None
        assert row.source_identities == (graph.semantic_identity,)
        assert row.resources == estimate_resources(row.target.executable)
        assert row.resources.used_qubits <= row.resources.n_qubits
        assert (
            row.resources.n_qubits
            == sum(len(v) for v in result.registers.values())
            + row.arithmetic.peak_workspace
        )
        assert row.construction_seconds >= 0 and row.target.elapsed_seconds >= 0
        assert row.stage_resources[-1] == ("executable", row.resources)
        for conversion in row.target.conversions:
            assert (
                conversion.source_program.content_hash == conversion.source_content_hash
            )
            assert conversion.program.content_hash == conversion.result_content_hash
        if target.basis:
            assert set(row.resources.operation_counts) <= set(target.basis)
    chosen = result.selected
    c = chosen.resources
    assert c is not None and chosen.arithmetic is not None
    costs = (c.n_qubits, c.depth, c.n_operations, chosen.arithmetic.peak_workspace)
    for row in result.rows:
        r = row.resources
        assert r is not None and row.arithmetic is not None
        other = (r.n_qubits, r.depth, r.n_operations, row.arithmetic.peak_workspace)
        assert not (
            all(a <= b for a, b in zip(other, costs, strict=True)) and other != costs
        )


def test_constraints_and_objectives_select_real_different_schedules() -> None:
    narrow = _select(constraints=Constraints(CircuitLimits(max_qubits=17)))
    shallow = _select(constraints=Constraints(CircuitLimits(max_operations=95)))
    assert narrow.selected is not None and shallow.selected is not None
    assert narrow.selected.strategy == "recompute"
    assert shallow.selected.strategy == "retain"
    depth = _select(objective="depth")
    assert depth.selected is not None and depth.selected.strategy == "retain"
    bounded = _select(constraints=Constraints(max_workspace=12))
    assert bounded.selected is not None and bounded.selected.strategy == "recompute"
    assert _select().ranking_version == 1
    with pytest.raises(TypeError):
        invalid_mutation: Any = narrow.registers
        invalid_mutation["x"] = (7,)


def test_determinism_ignores_elapsed_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.compiler._construction.scheduling.selection as selection

    first = _select()
    monkeypatch.setattr(selection, "perf_counter", lambda: 1e9)
    second = _select()
    assert first.selected is not None and second.selected is not None
    assert first.selected.strategy == second.selected.strategy
    assert all(row.target is not None for row in (*first.rows, *second.rows))
    assert [r.target.executable.content_hash for r in first.rows if r.target] == [
        r.target.executable.content_hash for r in second.rows if r.target
    ]


def test_all_violations_and_every_combined_recovery_is_retried() -> None:
    constraints = Constraints(
        CircuitLimits(max_qubits=16, max_operations=94, max_depth=68), max_workspace=11
    )
    failed = _select(constraints=constraints)
    assert failed.selected is None and failed.complete
    assert failed.diagnostics[0].code == "selection.infeasible"
    assert len(failed.recoveries) == 2
    for row, recovery in zip(failed.rows, failed.recoveries, strict=True):
        assert {d.code.rsplit(".", 1)[-1] for d in row.diagnostics} == {
            "max_qubits",
            "max_operations",
            "max_depth",
            "max_workspace",
        }
        retry = _select(constraints=recovery.constraints)
        assert retry.selected is not None
        assert next(r for r in retry.rows if r.strategy == recovery.strategy).feasible
        assert "not a global lower bound" in recovery.message
    # A width-only relaxation cannot repair this combination.
    assert (
        _select(
            constraints=Constraints(replace(constraints.limits, max_qubits=18), 11)
        ).selected
        is None
    )


@pytest.mark.parametrize(
    "budget",
    [
        ConstructionBudget(candidate_checks=1),
        ConstructionBudget(qubits=17),
        ConstructionBudget(operations=105),
        ConstructionBudget(qubits=3),
    ],
)
def test_incomplete_budget_never_selects_best_so_far(
    budget: ConstructionBudget,
) -> None:
    failed = _select(budget=budget)
    assert not failed.complete and failed.selected is None
    assert not failed.recoveries and len(failed.rows) == 2
    assert failed.diagnostics[0].code == "selection.budget"
    retry = _select()
    assert retry.complete and retry.selected is not None


@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_discrete_workspace_rejects_nonintegers(value: Any) -> None:
    with pytest.raises(ValueError):
        Constraints(max_workspace=value)


@pytest.mark.parametrize(
    "kwargs",
    [{"qubits": True}, {"qubits": 257}, {"operations": 0}, {"candidate_checks": 3}],
)
def test_construction_budgets_are_distinct_bounded_discrete_values(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        ConstructionBudget(**kwargs)


def test_preparation_scope_final_limit_does_not_prune_larger_source() -> None:
    # 100 cancelling input X gates consume construction budget but no final cost.
    preparation = CircuitIR(2, tuple(Instruction("x", (0,)) for _ in range(100)))
    result = _select(
        preparation=preparation,
        constraints=Constraints(CircuitLimits(max_operations=95)),
    )
    assert result.selected is not None and result.selected.strategy == "retain"
    costs = dict(result.selected.stage_resources)
    assert costs["source"].n_operations == 205
    assert costs["executable"].n_operations == 95


def test_preparation_cannot_claim_dirty_workspace_or_measurements() -> None:
    for preparation in (
        CircuitIR(6, ()),
        CircuitIR(2, (), measurements=(MeasurementNode("probabilities", (0,)),)),
    ):
        with pytest.raises(ValueError, match="input/output"):
            _select(preparation=preparation)


def test_unknown_basis_and_valid_conversion_gap_have_distinct_failures() -> None:
    with pytest.raises(BasisConversionError, match="unknown opcode"):
        Target(basis=("made_up",))
    gap = _select(
        target=Target(basis=("cx",)),
        preparation=CircuitIR(2, (Instruction("h", (0,)),)),
    )
    assert gap.complete and gap.selected is None
    assert all(r.diagnostics[0].code == "target.unsupported" for r in gap.rows)
    assert not gap.recoveries


def test_actual_normalization_basis_drift_is_not_silently_repaired() -> None:
    result = _select(
        target=Target(basis=("x", "cx", "ccx", "h", "s"), optimize=False),
        preparation=CircuitIR(2, (Instruction("h", (0,)), Instruction("s", (0,)))),
    )
    assert result.selected is None
    assert all(r.diagnostics[0].code == "target.basis_drift" for r in result.rows)


def test_explicit_topologies_are_not_resized_or_substituted() -> None:
    result = _select(
        target=Target(basis=BASIS, couplings=(CouplingMap(16, ((0, 1),)),))
    )
    assert result.complete and result.selected is None
    assert all("no explicit coupling" in r.diagnostics[0].message for r in result.rows)
    with pytest.raises(ValueError, match="unambiguous"):
        Target(couplings=(CouplingMap(2, ((0, 1),)), CouplingMap(2, ((0, 1),))))


def test_target_transform_replacement_uses_same_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.compiler._construction.scheduling.selection as selection

    recorded: dict[str, TargetProduct] = {}

    def live(source: CircuitIR, target: Target, max_operations: int) -> TargetProduct:
        result = compile_target(source, target, max_operations)
        recorded[source.content_hash] = result
        return result

    monkeypatch.setattr(selection, "compile_target", live)
    first = _select()

    def replay(source: CircuitIR, target: Target, max_operations: int) -> TargetProduct:
        assert max_operations == 100_000 and target == Target()
        return recorded[source.content_hash]

    monkeypatch.setattr(selection, "compile_target", replay)
    second = _select()
    assert [r.resources for r in first.rows] == [r.resources for r in second.rows]
    assert first.selected is not None and second.selected is not None
    assert first.selected.strategy == second.selected.strategy


def test_routed_request_is_deterministic_even_after_topology_cache_use() -> None:
    topology = Target(
        basis=BASIS,
        couplings=tuple(
            CouplingMap(n, tuple((i, i + 1) for i in range(n - 1))) for n in (17, 18)
        ),
        optimize=False,
        restore_layout=False,
    )
    first = _select(target=topology)
    for coupling in topology.couplings:
        coupling.shortest_path(0, coupling.n_qubits - 1)
    second = _select(target=topology)
    assert first.selected is not None and second.selected is not None
    assert all(row.target is not None for row in (*first.rows, *second.rows))
    assert [r.target.content_identities for r in first.rows if r.target] == [
        r.target.content_identities for r in second.rows if r.target
    ]


def test_stale_candidate_stage_is_refused_without_recompiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.compiler._construction.scheduling.target as target_module
    from flagquantum.compiler._construction.scheduling.target import TargetError

    report = _select()
    assert report.selected is not None and report.selected.target is not None
    product = report.selected.target

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("validation must not compile")

    monkeypatch.setattr(target_module, "compile", forbidden)
    product.validate()
    metadata = product.submitted.metadata
    assert isinstance(metadata, dict)
    metadata["changed_after_selection"] = True
    with pytest.raises(TargetError, match="changed"):
        product.validate()


def test_target_validation_reads_each_shared_object_once_and_detects_later_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.compiler._construction.scheduling.target import TargetError
    from flagquantum.compiler.layout import Layout

    program = CircuitIR(1, (Instruction("h", (0,)),))
    descriptor = vars(CircuitIR)["content_hash"]
    assert isinstance(descriptor, property)
    original = descriptor.fget
    assert original is not None
    calls: list[int] = []

    def digest(value: CircuitIR) -> str:
        calls.append(id(value))
        identity = original(value)
        assert isinstance(identity, str)
        return identity

    monkeypatch.setattr(CircuitIR, "content_hash", property(digest))
    target = TargetProduct(
        program,
        program,
        tuple(
            (name, program)
            for name in ("source", "submitted", "normalized", "executable")
        ),
        (),
        Layout(range(1)),
        None,
        0.0,
    )
    assert calls == [id(program)]
    target.validate()
    assert calls == [id(program)] * 2
    assert isinstance(program.metadata, dict)
    program.metadata["changed"] = True
    with pytest.raises(TargetError, match="changed"):
        target.validate()
    assert calls == [id(program)] * 3


def test_intermediate_conversion_budget_is_not_reported_as_infeasibility() -> None:
    report = _select(
        target=Target(basis=BASIS), budget=ConstructionBudget(operations=200)
    )
    assert not report.complete and report.selected is None
    assert any(row.diagnostics[0].code == "target.budget" for row in report.rows)
    assert _select(target=Target(basis=BASIS)).selected is not None


def test_conditioned_and_symbolic_preparations_fail_before_candidate_construction() -> (
    None
):
    from flagquantum.compiler._construction.scheduling.target import TargetError
    from flagquantum.core.parameters import Parameter

    for instruction in (
        Instruction("x", (0,), metadata={"conditions": ((0, 1),)}),
        Instruction("ry", (0,), params={"theta": Parameter("theta")}),
    ):
        with pytest.raises(ValueError, match="static|bind symbolic"):
            _select(preparation=CircuitIR(2, (instruction,)))
    with pytest.raises(TargetError, match="physical couplings"):
        compile_target(
            CircuitIR(3, (Instruction("ccx", (0, 1, 2)),)),
            Target(couplings=(CouplingMap(3, ((0, 1), (1, 2))),), optimize=False),
            100_000,
        )


def test_selection_performs_no_quantum_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum as fq

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Compiler candidate search cannot plan or execute")

    monkeypatch.setattr(fq, "plan", forbidden)
    monkeypatch.setattr(fq, "run", forbidden)
    assert _select().selected is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"optimize": 1},
        {"restore_layout": None},
        {"config": None},
        {"couplings": (None,)},
    ],
)
def test_target_controls_reject_ambiguous_or_invalid_types(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(TypeError):
        Target(**kwargs)


def test_discrete_controls_and_unadmitted_preparation_shapes_fail_early() -> None:
    import torch

    import flagquantum as fq
    from flagquantum.core.runtime_config import RuntimeConfig

    with pytest.raises(ValueError, match="local CPU"):
        Target(config=RuntimeConfig(device="cuda"))
    with pytest.raises(TypeError, match="shared"):
        invalid_limits: Any = {"max_qubits": 9}
        Constraints(limits=invalid_limits)
    with pytest.raises(TypeError, match="Constraints"):
        _select(constraints={"max_qubits": 9})
    with pytest.raises(ValueError, match="objective"):
        _select(objective="fastest")
    with pytest.raises(ValueError):
        invalid_workspace: Any = torch.tensor(5)
        Constraints(max_workspace=invalid_workspace)
    with pytest.raises(TypeError, match="explicit CircuitIR"):
        _select(preparation=fq.Circuit(2))
    with pytest.raises(ValueError, match="input/output"):
        _select(preparation=fq.Circuit(2, bsz=2).to_ir())


def test_one_qubit_double_cost_advisory_includes_blocked_alternative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.compiler._construction.scheduling.selection as selection

    # Synthetic estimator fixture exercises policy only, not numerical evidence
    # or a claim that these inflated costs are a real Runtime execution.
    original = estimate_resources

    def expensive_recomputation(program: CircuitIR) -> ResourceEstimate:
        costs = original(program)
        return (
            replace(costs, n_operations=250, depth=200)
            if costs.n_qubits == 17
            else costs
        )

    monkeypatch.setattr(selection, "estimate_resources", expensive_recomputation)
    report = _select(constraints=Constraints(CircuitLimits(max_qubits=17)))
    assert report.selected is not None and report.selected.strategy == "recompute"
    assert not report.rows[0].feasible
    assert any(
        d.code == "selection.tradeoff" and "retain" in d.message
        for d in report.diagnostics
    )


def test_internal_ssa_scopes_are_not_parameter_names() -> None:
    from flagquantum.compiler._construction.verification.profile import uint_type
    from flagquantum.compiler._hybrid.model import (
        Block,
        HybridProgram,
        Operation,
        Region,
        Value,
        ValueId,
    )

    x = Value(ValueId("entry", 0), uint_type(1))
    y = Value(ValueId("entry", 1), uint_type(1))
    z = Value(ValueId("entry", 2), uint_type(2))
    graph = HybridProgram(
        "sum",
        Region(
            (
                Block(
                    (x, y),
                    (
                        Operation("qint.add", (x, y), (z,)),
                        Operation("qint.return", (z,)),
                    ),
                ),
            )
        ),
    )
    anonymous = select_arithmetic((graph,), entry="sum")
    named = select_arithmetic((graph,), entry="sum", input_names=("x", "y"))
    assert anonymous.selected is not None and named.selected is not None
    assert anonymous.registers == {"arg0": (0,), "arg1": (1,), "result": (2, 3)}
    assert named.registers == {"x": (0,), "y": (1,), "result": (2, 3)}
    invalid_names: tuple[tuple[Any, ...], ...] = (
        ("x",),
        ("x", "x"),
        ("x", "result"),
        ("x", 1),
    )
    for names in invalid_names:
        with pytest.raises(ValueError, match="input_names"):
            select_arithmetic((graph,), entry="sum", input_names=names)
    with pytest.raises(TypeError, match="input_names"):
        select_arithmetic((graph,), entry="sum", input_names="xy")
