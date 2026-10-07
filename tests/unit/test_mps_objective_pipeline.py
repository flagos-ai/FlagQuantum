"""Local numerical and preflight checks for objective pipeline slots."""

import pytest
import torch

from flagquantum.runtime.executors.mps import reverse_z_observables as objectives
from flagquantum.runtime.executors.mps.reverse_observables import (
    mps_expectation_and_adjoints,
)
from flagquantum.runtime.executors.mps.state import RankOwnedMPSState
from flagquantum.simulation.mps.models import MPSConfig

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("target", [None, 0.3])
def test_pauli_adjoint_matches_direct_tensor_calculation(
    monkeypatch: pytest.MonkeyPatch, axis: str, target: float | None
) -> None:
    def broadcast(tensor: torch.Tensor, *, src: int) -> None:
        assert src == 0

    monkeypatch.setattr(objectives.dist, "broadcast", broadcast)
    tensor = torch.tensor([[1.0, 0.5j], [0.3j, 0.7]], dtype=torch.complex128).reshape(
        2, 1, 2, 1
    )
    tensor.requires_grad_(True)
    state = RankOwnedMPSState(1, 2, 0, 1, MPSConfig(), {0: tensor}, ((0,),))
    matrix = torch.tensor(
        {
            "x": [[0, 1], [1, 0]],
            "y": [[0, -1j], [1j, 0]],
            "z": [[1, 0], [0, -1]],
        }[axis],
        dtype=torch.complex128,
    )
    vector = tensor.reshape(2, 2)
    values = torch.einsum("bi,ij,bj->b", vector.conj(), matrix, vector).real
    expected = values.mean() if target is None else (values - target).square().mean()
    (gradient,) = torch.autograd.grad(expected, tensor)
    actual, adjoints = mps_expectation_and_adjoints(state, {0: axis}, target)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(adjoints[0], gradient)


def test_two_site_observable_scan_walks_a_chain_whose_bonds_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reverse scan must survive the shape its own first step presents.

    The walk starts at the rightmost site with the trivial environment
    ``(batch, 1, 1)``, so the first tensor it reaches is ``(chi, physical, 1)``:
    the two bond legs differ whenever the chain carries any correlation at all.
    A transfer that fuses a bond leg with the physical leg and folds it against
    the environment therefore has to say which leg it fused, and this test drives
    the real scan rather than the transfer in isolation so that a caller which
    hands it a square tensor cannot hide the difference. The dense expectation
    below is computed straight from the two site tensors, so it does not share
    the scan's contraction order.
    """

    def broadcast(tensor: torch.Tensor, *, src: int) -> None:
        assert src == 0

    monkeypatch.setattr(objectives.dist, "broadcast", broadcast)
    generator = torch.Generator().manual_seed(29)
    left = torch.randn(1, 1, 2, 2, generator=generator, dtype=torch.complex128)
    right = torch.randn(1, 2, 2, 1, generator=generator, dtype=torch.complex128)
    left.requires_grad_(True)
    right.requires_grad_(True)
    state = RankOwnedMPSState(2, 1, 0, 1, MPSConfig(), {0: left, 1: right}, ((0, 1),))
    operator = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

    # The chain as a dense state, then Z on wire 0 against it.
    dense = torch.einsum("apb,bqc->pqac", left[0], right[0]).reshape(4)
    operator = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    identity = torch.eye(2, dtype=torch.complex128)
    expected = torch.real(torch.vdot(dense, torch.kron(operator, identity) @ dense))
    expected_left, expected_right = torch.autograd.grad(expected, (left, right))

    actual, adjoints = mps_expectation_and_adjoints(state, {0: "z"})
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(adjoints[0], expected_left)
    torch.testing.assert_close(adjoints[1], expected_right)


def test_parameterized_observable_is_rejected_before_scan() -> None:
    tensor = torch.ones(1, 1, 2, 1, dtype=torch.complex128)
    state = RankOwnedMPSState(1, 1, 0, 1, MPSConfig(), {0: tensor}, ((0,),))
    with pytest.raises(ValueError, match="requires a fixed gate matrix"):
        mps_expectation_and_adjoints(state, {0: "rx"})


def test_pipeline_preserves_slot_values_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broadcast(tensor: torch.Tensor, *, src: int) -> None:
        assert src == 0

    def all_reduce(tensor: torch.Tensor, *, op: object) -> None:
        assert tensor.numel() == 1

    monkeypatch.setattr(objectives.dist, "broadcast", broadcast)
    monkeypatch.setattr(objectives.dist, "all_reduce", all_reduce)
    generator = torch.Generator().manual_seed(17)
    states = []
    for _ in range(3):
        tensors = {}
        for wire in range(2):
            tensor = torch.randn(
                2, 1, 2, 1, generator=generator, dtype=torch.complex128
            )
            tensors[wire] = tensor / torch.linalg.vector_norm(
                tensor, dim=2, keepdim=True
            )
        states.append(RankOwnedMPSState(2, 2, 0, 1, MPSConfig(), tensors, ((0, 1),)))
    terms = [(({0: "z"}, 0.1 * slot), ({0: "z", 1: "z"}, -0.2)) for slot in range(3)]
    expected = [
        objectives.mps_multi_observable_mse_and_adjoints(state, slot_terms)
        for state, slot_terms in zip(states, terms, strict=True)
    ]
    drained = []
    actual = objectives.site_sharded_z_zz_objective_pipeline(
        states, terms, max_pipeline_slots=2, on_window_drained=drained.append
    )
    assert drained == [2, 3]
    assert len(actual) == len(expected)
    for (loss, adjoints), (expected_loss, expected_adjoints) in zip(
        actual, expected, strict=True
    ):
        torch.testing.assert_close(loss, expected_loss)
        assert adjoints.keys() == expected_adjoints.keys()
        for wire, gradient in adjoints.items():
            torch.testing.assert_close(gradient, expected_adjoints[wire])

    with pytest.raises(ValueError, match="only Z and adjacent-ZZ"):
        objectives.site_sharded_z_zz_objective_pipeline(
            states[:2], (terms[0], (({0: "x"}, 0.0),))
        )
