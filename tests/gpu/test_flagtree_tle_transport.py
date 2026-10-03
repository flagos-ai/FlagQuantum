"""Real-device correctness for FlagTree-owned TLE transport kernels."""

from __future__ import annotations

import pytest
import torch

from flagquantum.kernels.flagtree import (
    pack_complex64_control_one_tle,
    unpack_complex64_control_one_tle,
)
from flagquantum.kernels.provenance import triton_compiler_provenance

pytestmark = [pytest.mark.gpu, pytest.mark.triton]


def _require_flagtree_070() -> None:
    distribution, version, integration_path, status = triton_compiler_provenance()
    assert (distribution, integration_path, status) == (
        "flagtree",
        "flagtree",
        "resolved",
    )
    assert version == "0.7.0"


def _control_indices(
    *,
    bit_position: int,
    compressed_start: int,
    compressed_end: int,
) -> torch.Tensor:
    compressed = torch.arange(
        compressed_start,
        compressed_end,
        dtype=torch.long,
        device="cuda",
    )
    low = compressed & ((1 << bit_position) - 1)
    return ((compressed - low) << 1) | low | (1 << bit_position)


@pytest.mark.parametrize(
    ("bit_position", "compressed_start", "compressed_end"),
    [(0, 0, 512), (5, 13, 397), (9, 64, 512)],
)
def test_flagtree_tle_control_pack_matches_index_reference(
    bit_position: int,
    compressed_start: int,
    compressed_end: int,
) -> None:
    _require_flagtree_070()
    generator = torch.Generator(device="cuda").manual_seed(
        261006 + bit_position + compressed_start
    )
    state = torch.randn(
        2,
        1 << 10,
        dtype=torch.complex64,
        device="cuda",
        generator=generator,
    )
    indices = _control_indices(
        bit_position=bit_position,
        compressed_start=compressed_start,
        compressed_end=compressed_end,
    )

    actual = pack_complex64_control_one_tle(
        state,
        bit_position=bit_position,
        compressed_start=compressed_start,
        compressed_end=compressed_end,
    )

    torch.testing.assert_close(actual, state[:, indices])


@pytest.mark.parametrize(
    ("bit_position", "compressed_start", "compressed_end"),
    [(0, 0, 512), (5, 13, 397), (9, 64, 512)],
)
def test_flagtree_tle_control_unpack_matches_index_reference(
    bit_position: int,
    compressed_start: int,
    compressed_end: int,
) -> None:
    _require_flagtree_070()
    generator = torch.Generator(device="cuda").manual_seed(
        261106 + bit_position + compressed_start
    )
    packed = torch.randn(
        2,
        compressed_end - compressed_start,
        dtype=torch.complex64,
        device="cuda",
        generator=generator,
    )
    output = torch.randn(
        2,
        1 << 10,
        dtype=torch.complex64,
        device="cuda",
        generator=generator,
    )
    expected = output.clone()
    indices = _control_indices(
        bit_position=bit_position,
        compressed_start=compressed_start,
        compressed_end=compressed_end,
    )
    expected[:, indices] = packed

    unpack_complex64_control_one_tle(
        packed,
        output,
        bit_position=bit_position,
        compressed_start=compressed_start,
    )

    torch.testing.assert_close(output, expected)
