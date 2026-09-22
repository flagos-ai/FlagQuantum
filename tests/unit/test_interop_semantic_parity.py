from __future__ import annotations

import pytest
import torch

from flagquantum.ecosystem._semantic_parity import (
    maximum_error_up_to_global_phase,
    reference_state,
    semantic_parity_cases,
)

pytestmark = pytest.mark.unit


def test_shared_semantic_corpus_is_deterministic_and_exercises_every_wire() -> None:
    first = semantic_parity_cases()
    second = semantic_parity_cases()

    assert [case.name for case in first] == [case.name for case in second]
    assert [case.program.to_dict() for case in first] == [
        case.program.to_dict() for case in second
    ]
    for case in first:
        used_wires = {
            wire
            for instruction in case.program.instructions
            for wire in instruction.wires
        }
        assert used_wires == set(range(case.program.n_wires))
        assert reference_state(case).dtype == torch.complex128


def test_state_error_ignores_global_phase_but_not_relative_phase() -> None:
    expected = torch.tensor([1, 1j], dtype=torch.complex128) / 2**0.5
    phased = expected * torch.exp(torch.tensor(0.731j, dtype=torch.complex128))
    changed = phased.clone()
    changed[1] *= -1

    assert maximum_error_up_to_global_phase(phased, expected) < 1e-12
    assert maximum_error_up_to_global_phase(changed, expected) > 1


def test_state_error_rejects_different_shapes() -> None:
    with pytest.raises(ValueError, match="statevector shapes differ"):
        maximum_error_up_to_global_phase(torch.zeros(2), torch.zeros(4))
