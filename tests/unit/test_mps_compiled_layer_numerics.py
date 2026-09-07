import torch

from flagquantum.core.ir import Instruction
from flagquantum.simulation.mps.models import MPSConfig
from flagquantum.simulation.mps_compiled_layers import (
    apply_compiled_mps_one_site_bucket,
    apply_compiled_mps_two_site_bucket,
    apply_mps_one_site_bucket,
)
from flagquantum.simulation.mps_rank_local import apply_rank_local_mps_instruction


def _site(bit: int) -> torch.Tensor:
    tensor = torch.zeros((1, 1, 2, 1), dtype=torch.complex128)
    tensor[0, 0, bit, 0] = 1
    return tensor


def test_compiled_one_site_bucket_matches_independent_rank_local_steps() -> None:
    instructions = (
        Instruction("ry", (0,), params={"theta": 0.25}),
        Instruction("ry", (2,), params={"theta": -0.75}),
    )
    tensors = (_site(0), _site(1))

    actual = apply_compiled_mps_one_site_bucket(
        instructions,
        tensors,
        bsz=1,
        device="cpu",
        dtype=torch.complex128,
        compiled=False,
    )

    for instruction, tensor, output in zip(instructions, tensors, actual):
        (expected,), split = apply_rank_local_mps_instruction(
            instruction,
            (tensor,),
            MPSConfig(),
            bsz=1,
            device="cpu",
            dtype=torch.complex128,
        )
        torch.testing.assert_close(output, expected)
        assert split is None


def test_mixed_one_site_bucket_matches_independent_rank_local_steps() -> None:
    instructions = (Instruction("x", (0,)), Instruction("h", (2,)))
    tensors = (_site(0), _site(1))

    actual = apply_mps_one_site_bucket(
        instructions,
        tensors,
        bsz=1,
        device="cpu",
        dtype=torch.complex128,
        compile_ry=True,
    )

    for instruction, tensor, output in zip(instructions, tensors, actual):
        (expected,), _ = apply_rank_local_mps_instruction(
            instruction,
            (tensor,),
            MPSConfig(),
            bsz=1,
            device="cpu",
            dtype=torch.complex128,
        )
        torch.testing.assert_close(output, expected)


def test_compiled_two_site_bucket_matches_independent_rank_local_steps() -> None:
    instructions = (
        Instruction("rxx", (0, 1), params={"theta": 0.2}),
        Instruction("ryy", (2, 3), params={"theta": -0.4}),
    )
    lefts = (_site(0), _site(1))
    rights = (_site(0), _site(1))
    config = MPSConfig()

    actual = apply_compiled_mps_two_site_bucket(
        instructions,
        lefts,
        rights,
        config,
        bsz=1,
        device="cpu",
        dtype=torch.complex128,
        isolate_factorizations=False,
        svd_driver=None,
        compiled=False,
    )

    for instruction, left, right, output in zip(instructions, lefts, rights, actual):
        expected_tensors, expected_info = apply_rank_local_mps_instruction(
            instruction,
            (left, right),
            config,
            bsz=1,
            device="cpu",
            dtype=torch.complex128,
        )
        actual_state = torch.einsum("blsm,bmtr->blstr", output[0], output[1])
        expected_state = torch.einsum(
            "blsm,bmtr->blstr", expected_tensors[0], expected_tensors[1]
        )
        torch.testing.assert_close(actual_state, expected_state)
        assert output[2]["discarded_weight"] == expected_info["discarded_weight"]
