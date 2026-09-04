import pytest
import torch

from flagquantum.core.ir import Instruction
from flagquantum.simulation.mps import MPSConfig
from flagquantum.simulation.mps_rank_local import apply_rank_local_mps_instruction

pytestmark = pytest.mark.unit


def _site(bit: int) -> torch.Tensor:
    tensor = torch.zeros((1, 1, 2, 1), dtype=torch.complex128)
    tensor[0, 0, bit, 0] = 1
    return tensor


def test_rank_local_instruction_path_applies_one_and_two_site_gates() -> None:
    (flipped,), split = apply_rank_local_mps_instruction(
        Instruction("x", (0,)),
        (_site(0),),
        MPSConfig(),
        bsz=1,
        device="cpu",
        dtype=torch.complex128,
    )
    torch.testing.assert_close(flipped, _site(1))
    assert split is None

    (left, right), split = apply_rank_local_mps_instruction(
        Instruction("cx", (0, 1)),
        (_site(1), _site(0)),
        MPSConfig(),
        bsz=1,
        device="cpu",
        dtype=torch.complex128,
    )
    state = torch.einsum("blsm,bmtr->blstr", left, right).reshape(1, 4)
    torch.testing.assert_close(
        state,
        torch.tensor([[0, 0, 0, 1]], dtype=torch.complex128),
    )
    assert split is not None
    assert split["discarded_weight"] == 0
