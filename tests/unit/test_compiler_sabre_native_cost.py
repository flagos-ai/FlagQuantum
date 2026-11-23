"""Route-preview semantics and the distinction between Weyl and delivered cost."""

from __future__ import annotations

import random
from dataclasses import replace
from functools import partial
from unittest.mock import patch

import pytest
import torch

from benchmarks import compiler_sabre_native_cost as native_cost
from benchmarks.compiler_sabre_native_cost import (
    _cost,
    _cz_lower_bound,
    _native_previews,
    _realize,
)
from flagquantum.compiler.lowering.basis_translation import translate
from flagquantum.compiler.topology import sabre
from flagquantum.compiler.topology.coupling import CouplingMap
from flagquantum.compiler.topology.sabre import plan_sabre_swaps
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.parameters import Parameter
from tests.unit.test_native_block_optimization import _operator, _stage

pytestmark = pytest.mark.unit


def _budget_case() -> CircuitIR:
    rng = random.Random(54139)
    instructions: list[Instruction] = []
    for _ in range(4):
        instructions.extend(
            (
                Instruction("cz", (0, 1)),
                Instruction("rx", (0,), {"theta": rng.uniform(-2, 2)}),
                Instruction("rz", (1,), {"theta": rng.uniform(-2, 2)}),
            )
        )
    return CircuitIR(2, tuple(instructions))


def _assert_equivalent(source: CircuitIR, output: CircuitIR) -> None:
    expected, actual = _operator(source), _operator(output)
    phase = torch.trace(actual.conj().T @ expected) / expected.shape[0]
    torch.testing.assert_close(expected, phase * actual, atol=1e-10, rtol=0)


@pytest.mark.parametrize("budget", (0, 256))
def test_weyl_estimate_cannot_override_the_actual_growth_budget(budget: int) -> None:
    source = _budget_case()
    coupling = CouplingMap.line(2)
    plan = plan_sabre_swaps(source, coupling)
    preview, routed_count = _native_previews(source, coupling)(plan)
    assert _cz_lower_bound(preview) == 2
    realized = _realize(preview, routed_count + budget)
    accepted, _ = _stage(source, budget=budget)
    assert _cost(realized) == _cost(accepted.program)
    assert len(realized.instructions) <= routed_count + budget
    assert _cost(realized)[0] == (4 if budget == 0 else 2)
    _assert_equivalent(source, realized)


def test_preview_preserves_reversed_control_and_restores_the_final_permutation() -> (
    None
):
    source = CircuitIR(
        4,
        (
            Instruction("rx", (0,), {"theta": 0.31}),
            Instruction("cx", (3, 0)),
            Instruction("rz", (3,), {"theta": -0.67}),
            Instruction("cx", (2, 0)),
            Instruction("cx", (0, 3)),
        ),
    )
    coupling = CouplingMap.line(4)
    plan = plan_sabre_swaps(source, coupling)
    assert plan.swaps
    preview, routed_count = _native_previews(source, coupling)(plan)
    assert all(
        len(item.wires) != 2 or coupling.has_edge(*item.wires)
        for item in preview.instructions
    )
    assert all(item.name in {"rx", "rz", "cz"} for item in preview.instructions)
    _assert_equivalent(source, preview)
    _assert_equivalent(source, _realize(preview, routed_count + 256))


@pytest.mark.parametrize(
    "unsupported", ("annotation", "symbolic", "batched", "trainable")
)
def test_preview_refuses_source_semantics_it_does_not_model(unsupported: str) -> None:
    item = Instruction("rx", (0,), {"theta": 0.4})
    if unsupported == "annotation":
        item = replace(item, metadata={"label": "retained boundary"})
    elif unsupported == "symbolic":
        item = replace(item, params={"theta": Parameter("theta")})
    else:
        angle = (
            torch.tensor([0.2, 0.4])
            if unsupported == "batched"
            else torch.tensor(0.4, requires_grad=True)
        )
        item = replace(item, params={"theta": angle})
    source = CircuitIR(2, (item, Instruction("cz", (0, 1))))
    coupling = CouplingMap.line(2)
    with pytest.raises(ValueError, match="unannotated|symbolic"):
        _native_previews(source, coupling)(plan_sabre_swaps(source, coupling))


def test_preview_refuses_an_unmodeled_input_permutation() -> None:
    source = _budget_case()
    coupling = CouplingMap.line(2)
    plan = plan_sabre_swaps(source, coupling, initial_layout=(1, 0))
    with pytest.raises(ValueError, match="identity input placement"):
        _native_previews(source, coupling)(plan)


def test_preview_reuse_keeps_sources_and_topologies_isolated() -> None:
    source = CircuitIR(
        4, (Instruction("rx", (0,), {"theta": 0.31}), Instruction("cx", (3, 0)))
    )
    changed = replace(
        source,
        instructions=(
            replace(source.instructions[0], params={"theta": -0.84}),
            source.instructions[1],
        ),
    )
    for coupling in (CouplingMap.line(4), CouplingMap.ring(4)):
        preview = _native_previews(source, coupling)
        plan = plan_sabre_swaps(source, coupling)
        first, _ = preview(plan)
        repeated, _ = preview(plan)
        assert first == repeated
        _assert_equivalent(source, repeated)
        other, _ = _native_previews(changed, coupling)(
            plan_sabre_swaps(changed, coupling)
        )
        _assert_equivalent(changed, other)
        assert first != other


@pytest.mark.parametrize("budget", (2000, 20000))
def test_native_selection_reuses_search_and_keeps_delivered_guards(budget: int) -> None:
    from benchmarks.compiler_sabre_trial_reuse import _select

    rng = random.Random(3)
    source = CircuitIR(
        6,
        tuple(Instruction("cx", tuple(rng.sample(range(6), 2))) for _ in range(24)),
    )
    coupling = CouplingMap.line(6)
    incumbent = sabre.plan_sabre_swaps(source, coupling)
    with (
        patch.object(
            sabre, "_plan_sabre_trial", wraps=sabre._plan_sabre_trial
        ) as trial,
        patch.object(
            sabre, "_select_sabre_trials", partial(_select, native_budget=budget)
        ),
    ):
        selected = sabre.plan_sabre_swaps(source, coupling)
    assert trial.call_count == 5
    preview = _native_previews(source, coupling)
    before, before_count = preview(incumbent)
    after, after_count = preview(selected)
    realized = _realize(after, after_count + budget)
    assert all(
        new <= old
        for new, old in zip(
            _cost(realized), _cost(_realize(before, before_count + budget)), strict=True
        )
    )
    assert len(realized.instructions) <= after_count + budget
    _assert_equivalent(source, realized)


def test_reused_native_selector_keeps_custom_topology_objectives() -> None:
    from benchmarks.compiler_sabre_trial_reuse import _select

    source = CircuitIR(4, (Instruction("cx", (3, 0)), Instruction("cx", (1, 3))))
    coupling = CouplingMap.line(4)
    baseline = sabre._plan_sabre_trial(source, coupling)
    assert baseline is not None

    def objective(plan: sabre.SabrePlan) -> int:
        return sabre._restored_swap_count(plan, coupling) + len(plan.swaps)

    expected = sabre._select_sabre_trials(source, coupling, baseline, cost=objective)
    assert _select(source, coupling, baseline, cost=objective) == expected


@pytest.mark.parametrize("topology", ("line", "ring"))
def test_preview_templates_reuse_translation_across_trial_placements(
    topology: str,
) -> None:
    source = CircuitIR(
        4,
        (
            Instruction("h", (3,)),
            Instruction("cx", (3, 0)),
            Instruction("rzz", (2, 0), {"theta": 0.29}),
            Instruction("cx", (0, 3)),
            Instruction("crz", (1, 3), {"theta": -0.43}),
        ),
    )
    coupling = CouplingMap.line(4) if topology == "line" else CouplingMap.ring(4)
    plans = [
        sabre._plan_sabre_trial(source, coupling, tie_rng=random.Random(seed))
        for seed in range(5)
    ]
    assert all(plan is not None for plan in plans)
    placements: set[tuple[tuple[int, ...], ...]] = set()
    with patch.object(native_cost, "translate", wraps=translate) as lowering:
        preview = _native_previews(source, coupling)
        for plan in plans:
            assert plan is not None
            placements.add(plan.placements)
            actual, count = preview(plan)
            assert count >= len(source.instructions)
            _assert_equivalent(source, actual)
            assert preview(plan) == (actual, count)
            assert all(
                len(gate.wires) != 2 or coupling.has_edge(*gate.wires)
                for gate in actual.instructions
            )
        assert len(placements) > 1
        translated = [call.args[0] for call in lowering.call_args_list]
        assert sorted((gate.name, gate.wires) for gate in translated) == sorted(
            (gate.name, gate.wires)
            for gate in (*source.instructions, Instruction("swap", (0, 1)))
        )
        assert lowering.call_count == len(source.instructions) + 1
        first = plans[0]
        assert first is not None
        _native_previews(source, coupling)(first)
        assert lowering.call_count == 2 * (len(source.instructions) + 1)


def test_preview_rejects_translation_that_introduces_workspace_wires() -> None:
    source = CircuitIR(4, (Instruction("cx", (3, 0)),))
    coupling = CouplingMap.line(4)
    plan = plan_sabre_swaps(source, coupling)
    with (
        patch.object(
            native_cost, "translate", return_value=(Instruction("cz", (1, 2)),)
        ),
        pytest.raises(ValueError, match="workspace wires"),
    ):
        _native_previews(source, coupling)(plan)
