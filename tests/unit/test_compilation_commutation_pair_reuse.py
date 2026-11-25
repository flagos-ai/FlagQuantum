"""Structural reuse must preserve partitions, counters and refusal boundaries."""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest
import torch

from flagquantum.compiler.optimization import commutation
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def _analyze(program: CircuitIR, limit: int) -> commutation.CommutationAnalysis:
    with patch.object(commutation, "_COMMUTATION_PAIR_CACHE_LIMIT", limit):
        return commutation.analyze_commutation(program)


def test_repeated_layers_reuse_both_acceptance_and_refusal_verdicts() -> None:
    program = CircuitIR(
        2,
        tuple(
            gate
            for _ in range(40)
            for gate in (
                Instruction("rz", (0,), {"theta": 0.3}),
                Instruction("cx", (0, 1)),
                Instruction("h", (0,)),
            )
        ),
    )
    predicate = commutation.commute
    with patch.object(commutation, "commute", wraps=predicate) as uncached:
        expected = _analyze(program, 0)
    with patch.object(commutation, "commute", wraps=predicate) as cached:
        actual = _analyze(program, 512)
    assert actual == expected
    assert cached.call_count < uncached.call_count / 4
    assert actual.queried_pairs == expected.queried_pairs
    assert actual.commuting_pairs == expected.commuting_pairs


@pytest.mark.parametrize("limit", [0, 1, 7, 512])
def test_cache_saturation_preserves_seeded_program_partitions(limit: int) -> None:
    rng = random.Random(917)
    gates = [
        Instruction("h", (0,)),
        Instruction("x", (1,)),
        Instruction("cx", (0, 1)),
        Instruction("cx", (1, 0)),
        Instruction("cz", (1, 2)),
        Instruction("rz", (0,), {"theta": 0.7}),
        Instruction("rx", (1,), {"theta": torch.tensor(0.2, requires_grad=True)}),
        Instruction("u3", (2,), {"theta": 0.2, "phi": 0.3, "lbd": 0.4}),
        Instruction("h", (0,), matrix=[[0, 1], [1, 0]]),
    ]
    for _ in range(20):
        program = CircuitIR(3, tuple(rng.choice(gates) for _ in range(80)))
        assert _analyze(program, limit) == _analyze(program, 0)


def test_later_calls_reinspect_mutable_parameter_presence_and_matrix_override() -> None:
    left, right = Instruction("h", (0,)), Instruction("h", (0,))
    program = CircuitIR(1, (left, right))
    assert commutation.analyze_commutation(program).commuting_pairs == 1
    left.params["diagnostic"] = 1
    assert _analyze(program, 512) == _analyze(program, 0)
    assert commutation.analyze_commutation(program).commuting_pairs == 0
    overridden = CircuitIR(1, (Instruction("h", (0,), matrix=[[0, 1], [1, 0]]), right))
    assert commutation.analyze_commutation(overridden).commuting_pairs == 0


def test_replacement_predicate_is_observed_by_the_next_sweep() -> None:
    program = CircuitIR(1, (Instruction("h", (0,)), Instruction("h", (0,))))
    assert commutation.analyze_commutation(program).commuting_pairs == 1
    with patch.object(commutation, "commute", return_value=False):
        assert commutation.analyze_commutation(program).commuting_pairs == 0
