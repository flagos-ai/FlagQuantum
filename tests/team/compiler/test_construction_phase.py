"""Closed phase binding, exact selected-prefix adjoints and resource accounting."""

from dataclasses import replace

import pytest

from flagquantum.compiler._construction.bindings import verify_bindings
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lowering import schedule_arithmetic
from flagquantum.compiler._construction.selection import (
    ConstructionBudget,
    select_arithmetic,
)
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def marked(x: int, y: int) -> bool:
    return x * x + y < 5


def wide(x: int) -> int:
    return x + 1


def bit(x: int) -> bool:
    return x


def bindings(graph):
    result = {}
    position = 0
    for value in graph.body.blocks[0].arguments:
        width = value.type.parameters[0]
        result[value.id] = tuple(range(position, position + width))
        position += width
    return result, position


def build(strategy="retain", *, max_operations=100_000):
    graph = capture(marked, {"x": 2, "y": 2})
    inputs, position = bindings(graph)
    return schedule_arithmetic(
        (graph,),
        inputs,
        (),
        entry=graph.name,
        n_qubits=64,
        workspace=tuple(range(position, 64)),
        workspace_state="fresh_zero",
        scheduling=strategy,
        output_mode="phase",
        max_operations=max_operations,
    )


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_literal_inverse_of_the_same_prefix_and_mirrored_allocation_events(strategy):
    candidate = build(strategy)
    gates = candidate.circuit.instructions
    middle = len(gates) // 2
    assert gates[middle].name == "z"
    assert gates[middle + 1 :] == tuple(reversed(gates[:middle]))
    assert {g.name for g in gates[:middle]} <= {"x", "cx", "ccx"}
    events = candidate.events
    split = len(events) // 2
    for before, after in zip(events[:split], reversed(events[split:]), strict=True):
        assert after == replace(
            before, action="uncompute" if before.action == "compute" else "compute"
        )
    assert sum(e.operation_count for e in events) + 1 == len(gates)
    live = set()
    computed = {}
    for e in events:
        if e.action == "compute":
            assert not live.intersection(e.allocation)
            live.update(e.allocation)
            computed[e.value] = computed.get(e.value, 0) + 1
        else:
            assert set(e.allocation) <= live
        assert not live.intersection(e.scratch)
        assert len(live) + len(e.scratch) == e.live_workspace
        if e.action == "uncompute":
            live.difference_update(e.allocation)
    assert not live
    assert candidate.peak_workspace == max(e.live_workspace for e in events)
    assert candidate.recomputations == sum(v - 1 for v in computed.values())


def test_direct_input_predicate_needs_no_work_or_value_target():
    graph = capture(bit, {"x": 1})
    inputs, _ = bindings(graph)
    verify_bindings(
        (graph,), inputs, (), entry=graph.name, n_wires=1, output_mode="phase"
    )
    result = schedule_arithmetic(
        (graph,), inputs, (), entry=graph.name, n_qubits=1, output_mode="phase"
    )
    assert result.circuit.instructions == (Instruction("z", (0,)),)
    assert result.peak_workspace == 0 and result.events == ()


def test_wide_return_value_target_unknown_mode_and_zero_obligations_are_refused():
    graph = capture(wide, {"x": 2})
    inputs, position = bindings(graph)
    with pytest.raises(HybridVerificationError, match="phase.predicate"):
        schedule_arithmetic(
            (graph,),
            inputs,
            (),
            entry=graph.name,
            n_qubits=16,
            workspace=tuple(range(position, 16)),
            workspace_state="fresh_zero",
            output_mode="phase",
        )
    graph = capture(bit, {"x": 1})
    inputs, _ = bindings(graph)
    for output, mode, workspace, state, reason in [
        ((1,), "phase", (), "unknown", "binding.phase"),
        ((), "invalid", (), "unknown", "output.mode"),
        ((), "phase", (1,), "unknown", "workspace.precondition"),
        ((), "phase", (0,), "fresh_zero", "binding.alias"),
    ]:
        with pytest.raises(HybridVerificationError, match=reason):
            verify_bindings(
                (graph,),
                inputs,
                output,
                entry=graph.name,
                n_wires=3,
                workspace=workspace,
                workspace_state=state,
                output_mode=mode,
            )


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_full_phase_and_inverse_count_against_the_existing_operation_budget(strategy):
    candidate = build(strategy)
    count = len(candidate.circuit.instructions)
    assert len(build(strategy, max_operations=count).circuit.instructions) == count
    with pytest.raises(HybridVerificationError, match="operations"):
        build(strategy, max_operations=count - 1)


@pytest.mark.parametrize("kind", ["non_self_adjoint", "annotated_gate"])
def test_an_unadmitted_prefix_gate_cannot_silently_use_a_wrong_adjoint(
    monkeypatch, kind
):
    from flagquantum.compiler._construction import lowering

    monkeypatch.setattr(
        lowering,
        "comparison_xor",
        lambda a, b, target, *args, **kwargs: (
            (
                Instruction("t", (target,))
                if kind == "non_self_adjoint"
                else Instruction("x", (target,), metadata={"is_dynamic": True})
            ),
        ),
    )
    with pytest.raises(HybridVerificationError, match="phase.adjoint"):
        build()


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_shared_selection_exposes_inputs_only_and_counts_every_phase_temporary(
    objective,
):
    graph = capture(marked, {"x": 2, "y": 2})
    selection = select_arithmetic(
        (graph,),
        entry=graph.name,
        input_names=("x", "y"),
        output_mode="phase",
        objective=objective,
    )
    assert selection.selected is not None and selection.complete
    assert dict(selection.registers) == {"x": (0, 1), "y": (2, 3)}
    for row in selection.rows:
        assert row.arithmetic is not None
        assert row.resources.n_qubits == 4 + row.arithmetic.peak_workspace
        assert row.resources == row.stage_resources[-1][1]
    assert len(selection.rows) == 2


def test_phase_preparation_cannot_touch_predicate_workspace_or_a_hidden_output():
    graph = capture(marked, {"x": 2, "y": 2})
    with pytest.raises(ValueError, match="input/output"):
        select_arithmetic(
            (graph,),
            entry=graph.name,
            output_mode="phase",
            preparation=CircuitIR(5, (Instruction("h", (4,)),)),
        )
    with pytest.raises(ValueError, match="mode"):
        select_arithmetic((graph,), entry=graph.name, output_mode="invalid")
    bad = capture(wide, {"x": 2})
    with pytest.raises(ValueError, match="one-bit"):
        select_arithmetic((bad,), entry=bad.name, output_mode="phase")
    incomplete = select_arithmetic(
        (graph,),
        entry=graph.name,
        output_mode="phase",
        budget=ConstructionBudget(candidate_checks=1),
    )
    assert not incomplete.complete and incomplete.selected is None


def test_projective_basis_paths_and_non_native_preparation_are_not_admitted():
    from flagquantum.compiler._construction.target import PHASE_BASIS, Target
    from flagquantum.compiler.coupling import CouplingMap

    graph = capture(marked, {"x": 2, "y": 2})
    for profile in (
        Target(couplings=(CouplingMap.line(32),)),
        Target(basis=PHASE_BASIS, optimize=True),
        Target(basis=("x", "cx", "u3"), optimize=False),
    ):
        with pytest.raises(ValueError, match="phase"):
            select_arithmetic(
                (graph,), entry=graph.name, output_mode="phase", target=profile
            )
    profile = Target(basis=PHASE_BASIS, optimize=False)
    for gate in (
        Instruction("y", (0,)),
        Instruction("x", (0,), metadata={"label": "unsupported"}),
    ):
        with pytest.raises(ValueError, match="plain native"):
            select_arithmetic(
                (graph,),
                entry=graph.name,
                output_mode="phase",
                target=profile,
                preparation=CircuitIR(4, (gate,)),
            )
