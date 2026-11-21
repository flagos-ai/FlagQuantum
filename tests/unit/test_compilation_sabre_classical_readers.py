"""Every classical reader must finish before a later measurement overwrites it."""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from flagquantum.compiler import CouplingMap, compile, route_to_topology
from flagquantum.compiler.topology import routing_strategy, sabre
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def _condition(key: str) -> dict[str, object]:
    value = ((0, 1),) if key == "conditions" else (((0, 1),),)
    return {"is_dynamic": True, key: value}


def _source(key: str) -> CircuitIR:
    return CircuitIR(
        6,
        (
            Instruction(
                "measure", (0,), metadata={"classical_bit": 0, "is_dynamic": True}
            ),
            Instruction("cx", (1, 5), metadata=_condition(key)),
            Instruction("x", (2,), metadata=_condition(key)),
            Instruction(
                "measure", (3,), metadata={"classical_bit": 0, "is_dynamic": True}
            ),
        ),
    )


def _classical_execution(
    program: CircuitIR, basis: int
) -> tuple[tuple[int, ...], dict[int, int]]:
    # Independent deterministic basis interpreter; it reads no planner graph.
    state = [(basis >> q) & 1 for q in range(program.n_wires)]
    memory: dict[int, int] = {}
    for instruction in program:
        clauses = instruction.metadata.get("condition_clauses")
        if clauses is None and "conditions" in instruction.metadata:
            clauses = (instruction.metadata["conditions"],)
        if clauses is not None and not any(
            all(memory[bit] == value for bit, value in clause) for clause in clauses
        ):
            continue
        wires = instruction.wires
        if instruction.name == "measure":
            memory[instruction.metadata["classical_bit"]] = state[wires[0]]
        elif instruction.name == "x":
            state[wires[0]] ^= 1
        elif instruction.name == "cx":
            state[wires[1]] ^= state[wires[0]]
        else:
            assert instruction.name == "swap"
            state[wires[0]], state[wires[1]] = state[wires[1]], state[wires[0]]
    return tuple(state), memory


@pytest.mark.parametrize("key", ["conditions", "condition_clauses"])
@pytest.mark.parametrize(
    "trial", ["deterministic", "seed-0", "seed-1", "seed-2", "layers"]
)
def test_each_trial_keeps_all_readers_before_the_overwrite(
    key: str, trial: str
) -> None:
    source = _source(key)
    coupling = CouplingMap.line(6)
    plan = sabre._plan_sabre_trial(
        source,
        coupling,
        tie_rng=random.Random(int(trial[-1])) if trial.startswith("seed-") else None,
        layer_lookahead=trial == "layers",
    )
    assert plan is not None
    assert plan.sequence.index(1) < plan.sequence.index(3)
    assert plan.sequence.index(2) < plan.sequence.index(3)
    with patch.object(
        routing_strategy, "plan_sabre_swaps", return_value=plan
    ) as planner:
        routed = route_to_topology(source, coupling, strategy="sabre")
    assert planner.call_count == 1
    assert routed.metadata["routing"]["mapping_restored"] is True
    for basis in range(1 << source.n_wires):
        assert _classical_execution(routed, basis) == _classical_execution(
            source, basis
        )


@pytest.mark.parametrize("key", ["conditions", "condition_clauses"])
def test_selected_route_preserves_the_complete_classical_truth_table(key: str) -> None:
    source = _source(key)
    routed = route_to_topology(source, CouplingMap.line(6), strategy="sabre")
    for basis in range(1 << source.n_wires):
        assert _classical_execution(routed, basis) == _classical_execution(
            source, basis
        )


def test_readers_are_tracked_again_after_a_second_write() -> None:
    source = CircuitIR(
        6,
        (
            *_source("conditions").instructions[:3],
            _source("conditions").instructions[3],
            Instruction("x", (4,), metadata=_condition("conditions")),
            Instruction("cx", (1, 5), metadata=_condition("conditions")),
            Instruction(
                "measure", (0,), metadata={"classical_bit": 0, "is_dynamic": True}
            ),
        ),
    )
    plan = sabre.plan_sabre_swaps(source, CouplingMap.line(6))
    assert sorted(plan.sequence) == list(range(len(source)))
    assert plan.sequence.index(1) < plan.sequence.index(3) < plan.sequence.index(4)
    assert plan.sequence.index(4) < plan.sequence.index(6)
    assert plan.sequence.index(5) < plan.sequence.index(6)


def test_unrelated_classical_bits_keep_independent_frontiers() -> None:
    source = CircuitIR(
        6,
        (
            *_source("conditions").instructions[:3],
            Instruction(
                "measure", (3,), metadata={"classical_bit": 1, "is_dynamic": True}
            ),
        ),
    )
    plan = sabre.plan_sabre_swaps(source, CouplingMap.line(6))
    assert plan.sequence.index(3) < plan.sequence.index(1)


@pytest.mark.parametrize("strategy", ["auto", "sabre"])
@pytest.mark.parametrize("preparation", ["x", "h"])
def test_measurement_feedback_runs_before_its_classical_bit_is_reused(
    strategy: str,
    preparation: str,
) -> None:
    import torch

    import flagquantum as fq
    from flagquantum.dynamic import DynamicCircuit

    source = CircuitIR(
        6,
        (
            Instruction(preparation, (0,)),
            Instruction("x", (1,)),
            *_source("conditions").instructions,
        ),
    )
    routed = compile(
        source,
        coupling_map=CouplingMap.line(6),
        routing_strategy=strategy,
        optimize=False,
    )
    expected = fq.experimental.dynamic.run_dynamic(
        DynamicCircuit.from_ir(source), shots=32, seed=3
    )
    actual = fq.experimental.dynamic.run_dynamic(
        DynamicCircuit.from_ir(routed), shots=32, seed=3
    )
    torch.testing.assert_close(actual.final_samples, expected.final_samples)
    torch.testing.assert_close(actual.classical_bits, expected.classical_bits)
    assert bool(torch.all(actual.final_samples[:, 5] == actual.final_samples[:, 0]))
    if preparation == "x":
        assert bool(torch.all(actual.final_samples[:, 5] == 1))
    else:
        assert set(actual.final_samples[:, 0].tolist()) == {0, 1}
    assert bool(torch.all(actual.classical_bits[:, 0] == 0))


def test_disjunction_reading_two_bits_preserves_both_overwrite_dependencies() -> None:
    source = CircuitIR(
        8,
        (
            Instruction(
                "measure", (0,), metadata={"classical_bit": 0, "is_dynamic": True}
            ),
            Instruction(
                "measure", (7,), metadata={"classical_bit": 1, "is_dynamic": True}
            ),
            Instruction(
                "cx",
                (1, 5),
                metadata={
                    "is_dynamic": True,
                    "condition_clauses": (
                        ((0, 1), (1, 0)),
                        ((0, 0), (1, 1)),
                    ),
                },
            ),
            Instruction("x", (2,), metadata=_condition("conditions")),
            Instruction(
                "x", (6,), metadata={"is_dynamic": True, "conditions": ((1, 1),)}
            ),
            Instruction(
                "measure", (3,), metadata={"classical_bit": 0, "is_dynamic": True}
            ),
            Instruction(
                "measure", (4,), metadata={"classical_bit": 1, "is_dynamic": True}
            ),
        ),
    )
    routed = route_to_topology(source, CouplingMap.line(8), strategy="sabre")
    for basis in range(1 << source.n_wires):
        assert _classical_execution(routed, basis) == _classical_execution(
            source, basis
        )
