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
    _preview_matrix_reader,
    _realize,
)
from flagquantum.compiler.optimization._cz_blocks import (
    _candidate,
    _numeric_matrix,
    _product,
)
from flagquantum.compiler.optimization.two_qubit_optimization import Matrix
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


@pytest.mark.parametrize("unsupported", ("annotation", "symbolic"))
def test_preview_refuses_source_semantics_it_does_not_model(unsupported: str) -> None:
    item = Instruction("rx", (0,), {"theta": 0.4})
    if unsupported == "annotation":
        item = replace(item, metadata={"label": "retained boundary"})
    else:
        item = replace(item, params={"theta": Parameter("theta")})
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


def test_matrix_reuse_preserves_ordered_basis_and_native_opcode() -> None:
    cases = (
        (Instruction("rx", (0,), {"theta": 0.31}), (0, 1)),
        (Instruction("rx", (3,), {"theta": 0.31}), (3, 7)),
        (Instruction("rx", (7,), {"theta": 0.31}), (3, 7)),
        (Instruction("rz", (7,), {"theta": 0.31}), (3, 7)),
        (Instruction("cz", (3, 7)), (3, 7)),
        (Instruction("cz", (7, 3)), (3, 7)),
    )
    expected = [_numeric_matrix(gate, pair) for gate, pair in cases]
    with patch.object(
        native_cost, "_numeric_matrix", wraps=_numeric_matrix
    ) as original:
        read = _preview_matrix_reader()
        for _ in range(2):
            assert [read(gate, pair) for gate, pair in cases] == expected
        assert original.call_count == 4


def test_matrix_reuse_reads_mutable_parameters_and_preserves_signed_zero() -> None:
    gate = Instruction("rx", (0,), {"theta": 0.0})
    assert isinstance(gate.params, dict)
    with patch.object(
        native_cost, "_numeric_matrix", wraps=_numeric_matrix
    ) as original:
        read = _preview_matrix_reader()
        for value in (0.0, -0.0, 0.31, -0.84, 0.31):
            gate.params["theta"] = value
            assert read(gate, (0, 1)) == _numeric_matrix(gate, (0, 1))
        assert original.call_count == 4


@pytest.mark.parametrize(
    "kind",
    (
        "tensor",
        "trainable",
        "batched",
        "symbolic",
        "extra",
        "annotation",
        "matrix",
        "outside",
    ),
)
def test_matrix_reuse_leaves_other_inputs_with_the_original_reader(kind: str) -> None:
    gate = Instruction("rx", (0,), {"theta": 0.31})
    if kind == "tensor":
        gate = replace(gate, params={"theta": torch.tensor(0.31)})
    elif kind == "trainable":
        gate = replace(gate, params={"theta": torch.tensor(0.31, requires_grad=True)})
    elif kind == "batched":
        gate = replace(gate, params={"theta": torch.tensor([0.31, 0.84])})
    elif kind == "symbolic":
        gate = replace(gate, params={"theta": Parameter("theta")})
    elif kind == "extra":
        gate = replace(gate, params={"theta": 0.31, "unused": Parameter("unused")})
    elif kind == "annotation":
        gate = replace(gate, metadata={"label": "retained boundary"})
    elif kind == "matrix":
        gate = replace(gate, matrix=torch.eye(2))
    else:
        gate = replace(gate, wires=(2,))
    expected = _numeric_matrix(gate, (0, 1))
    with patch.object(
        native_cost, "_numeric_matrix", wraps=_numeric_matrix
    ) as original:
        read = _preview_matrix_reader()
        assert read(gate, (0, 1)) == expected
        assert read(gate, (0, 1)) == expected
        assert original.call_count == 2


def test_matrix_reuse_has_bounded_storage_and_isolated_lifetimes() -> None:
    gates = [Instruction("rz", (0,), {"theta": float(i)}) for i in range(33)]
    with patch.object(
        native_cost, "_numeric_matrix", wraps=_numeric_matrix
    ) as original:
        read = _preview_matrix_reader()
        for gate in gates:
            assert read(gate, (0, 1)) == _numeric_matrix(gate, (0, 1))
        assert original.call_count == 33
        read(gates[0], (0, 1))
        assert original.call_count == 34
        read(gates[-1], (0, 1))
        assert original.call_count == 34
        _preview_matrix_reader()(gates[-1], (0, 1))
        assert original.call_count == 35


@pytest.mark.parametrize("budget", (0, 256))
def test_matrix_reuse_keeps_lower_bounds_realization_and_actual_budget(
    budget: int,
) -> None:
    source = _budget_case()
    read = _preview_matrix_reader()
    assert _cz_lower_bound(source, matrix_reader=read) == _cz_lower_bound(source)
    limit = len(source.instructions) + budget
    expected = _realize(source, limit)
    actual = _realize(source, limit, matrix_reader=read)
    assert actual == expected
    assert _cost(actual)[0] == (4 if budget == 0 else 2)
    _assert_equivalent(source, actual)


def _nonreducing_native_block() -> CircuitIR:
    return CircuitIR(
        2,
        (
            Instruction("cz", (0, 1)),
            Instruction("rx", (0,), {"theta": 0.31}),
            Instruction("rz", (1,), {"theta": 0.43}),
            Instruction("cz", (0, 1)),
            Instruction("rx", (0,), {"theta": 0.27}),
            Instruction("rz", (1,), {"theta": 0.66}),
        ),
    )


@pytest.mark.parametrize("budget", (0, 256))
@pytest.mark.parametrize("reverse", (False, True))
def test_known_nonreducing_cost_avoids_synthesis_without_changing_output(
    budget: int, reverse: bool
) -> None:
    source = _nonreducing_native_block()
    if reverse:
        source = replace(
            source,
            instructions=tuple(
                replace(item, wires=item.wires[::-1]) if len(item.wires) == 2 else item
                for item in source.instructions
            ),
        )
    costs: dict[bytes, int] = {}
    _cz_lower_bound(source, matrix_costs=costs)
    expected = _realize(source, len(source.instructions) + budget)
    with (
        patch.object(native_cost, "_candidate", wraps=_candidate) as synth,
        patch.object(
            native_cost,
            "_weyl_decomposition",
            side_effect=AssertionError("Unexpected extra classification"),
        ),
    ):
        actual = _realize(source, len(source.instructions) + budget, matrix_costs=costs)
    assert synth.call_count == 0
    assert actual == expected
    _assert_equivalent(source, actual)


@pytest.mark.parametrize("cache_state", ("absent", "empty", "unrelated"))
def test_unknown_cost_keeps_original_synthesis(cache_state: str) -> None:
    source = _nonreducing_native_block()
    costs = (
        None
        if cache_state == "absent"
        else ({b"unrelated": 3} if cache_state == "unrelated" else {})
    )
    with patch.object(native_cost, "_candidate", wraps=_candidate) as synth:
        actual = _realize(source, 256, matrix_costs=costs)
    assert synth.call_count == 1
    assert actual == _realize(source, 256)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("budget", (0, 256))
def test_reducing_blocks_still_synthesize_and_obey_growth(
    seed: int, budget: int
) -> None:
    rng = random.Random(seed)
    source = CircuitIR(
        2,
        tuple(
            item
            for _ in range(4)
            for item in (
                Instruction("cz", (0, 1)),
                Instruction("rx", (rng.randrange(2),), {"theta": rng.uniform(-2, 2)}),
                Instruction("rz", (rng.randrange(2),), {"theta": rng.uniform(-2, 2)}),
            )
        ),
    )
    costs: dict[bytes, int] = {}
    _cz_lower_bound(source, matrix_costs=costs)
    expected = _realize(source, len(source.instructions) + budget)
    with patch.object(native_cost, "_candidate", wraps=_candidate) as synth:
        actual = _realize(source, len(source.instructions) + budget, matrix_costs=costs)
    assert synth.call_count == 1
    assert actual == expected
    assert len(actual.instructions) <= len(source.instructions) + budget
    _assert_equivalent(source, actual)


def test_annotations_keep_original_candidate_path() -> None:
    source = _nonreducing_native_block()
    source = replace(
        source,
        instructions=tuple(
            replace(item, metadata={"label": "boundary"})
            for item in source.instructions
        ),
    )
    costs: dict[bytes, int] = {}
    _cz_lower_bound(source, matrix_costs=costs)
    with patch.object(native_cost, "_candidate", wraps=_candidate) as synth:
        actual = _realize(source, 256, matrix_costs=costs)
    assert synth.call_count == 1
    assert actual == _realize(source, 256)


def test_cost_table_cannot_hide_nonunitary_failure() -> None:
    source = _nonreducing_native_block()

    def nonunitary(item: Instruction, pair: tuple[int, int]) -> Matrix | None:
        result = _numeric_matrix(item, pair)
        return (
            None if result is None else [[2 * value for value in row] for row in result]
        )

    matrix = _product(source.instructions, (0, 1), matrix_reader=nonunitary)
    assert matrix is not None
    costs = {native_cost._matrix_key(matrix): 3}
    with (
        patch.object(native_cost, "_candidate", wraps=_candidate) as synth,
        pytest.raises(ValueError, match="matrix must be unitary"),
    ):
        _realize(source, 256, matrix_costs=costs, matrix_reader=nonunitary)
    assert synth.call_count == 1


def test_known_cost_does_not_override_initial_operation_budget() -> None:
    source = _nonreducing_native_block()
    costs: dict[bytes, int] = {}
    _cz_lower_bound(source, matrix_costs=costs)
    with pytest.raises(ValueError, match="Native preview exceeds the operation budget"):
        _realize(source, len(source.instructions) - 1, matrix_costs=costs)


@pytest.mark.parametrize("cz_count", (2, 3))
@pytest.mark.parametrize(
    "angle", (0.0, 1e-12, 0.31, 1.5707963267948966, 3.141592653589793)
)
def test_classification_reuse_preserves_gate_class_boundaries(
    cz_count: int, angle: float
) -> None:
    source = CircuitIR(
        2,
        tuple(
            item
            for index in range(cz_count)
            for item in (
                Instruction("cz", (0, 1) if index % 2 == 0 else (1, 0)),
                Instruction("rx", (index % 2,), {"theta": angle}),
                Instruction("rz", ((index + 1) % 2,), {"theta": -angle}),
            )
        ),
    )
    costs: dict[bytes, int] = {}
    _cz_lower_bound(source, matrix_costs=costs)
    for budget in (0, 256):
        expected = _realize(source, len(source.instructions) + budget)
        actual = _realize(source, len(source.instructions) + budget, matrix_costs=costs)
        assert actual == expected
        _assert_equivalent(source, actual)


def test_nonnative_block_keeps_translation_and_candidate_validation() -> None:
    source = _nonreducing_native_block()
    source = replace(
        source,
        instructions=tuple(
            replace(item, name="cx") if item.name == "cz" else item
            for item in source.instructions
        ),
    )
    costs: dict[bytes, int] = {}
    _cz_lower_bound(source, matrix_costs=costs)
    with patch.object(native_cost, "_candidate", wraps=_candidate) as synth:
        actual = _realize(source, 256, matrix_costs=costs)
    assert synth.call_count == 1
    assert actual == _realize(source, 256)
    _assert_equivalent(source, actual)
