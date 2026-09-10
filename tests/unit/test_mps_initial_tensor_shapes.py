"""Initial MPS shape reconciliation without a live process group."""

from unittest.mock import Mock

import pytest
import torch

from flagquantum.runtime.executors.mps.records import MPSReverseContractError
from flagquantum.runtime.executors.mps.state import normalize_rank_owned_initial_tensors

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("right_bond", [3, 4])
def test_initial_tensor_shapes_and_bond_validation(
    monkeypatch: pytest.MonkeyPatch, right_bond: int
) -> None:
    collective = Mock()
    monkeypatch.setattr(torch.distributed, "all_reduce", collective)
    tensors = {
        0: torch.ones((2, 1, 2, 3), dtype=torch.complex128),
        1: torch.ones((2, right_bond, 2, 1), dtype=torch.complex128),
    }

    def normalize() -> (
        tuple[dict[int, torch.Tensor], int, dict[int, tuple[int, int, int, int]]]
    ):
        return normalize_rank_owned_initial_tensors(
            tensors,
            ownership=((0, 1),),
            rank=0,
            world=1,
            n_wires=2,
            device=torch.device("cpu"),
            dtype=torch.complex128,
        )

    if right_bond != 3:
        with pytest.raises(MPSReverseContractError, match="initial MPS bond 0 differs"):
            normalize()
    else:
        local, batch_size, shapes = normalize()
        assert batch_size == 2
        assert shapes == {0: (2, 1, 2, 3), 1: (2, 3, 2, 1)}
        for wire, tensor in tensors.items():
            torch.testing.assert_close(local[wire], tensor)
            assert local[wire].data_ptr() != tensor.data_ptr()
    collective.assert_called_once()
