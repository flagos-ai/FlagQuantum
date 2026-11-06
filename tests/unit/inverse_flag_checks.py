"""Independent complete flag-channel certificates and raw phase projections."""

from itertools import product

import torch

from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch


def check_flag_channel(
    gates: tuple[Instruction, ...],
    inverse: tuple[Instruction, ...],
    outputs: tuple[int, ...],
    data_width: int,
    work_width: int,
    events: int,
    *,
    raw_branches: bool,
) -> None:
    """Check every logical column and optionally every unnormalized branch."""
    assert len(outputs) == 1 << data_width
    width = data_width + work_width
    before = tuple(
        tuple((x << work_width) >> (width - 1 - i) & 1 for i in range(width))
        for x in range(len(outputs))
    )
    after = tuple(
        tuple((x << work_width) >> (width - 1 - i) & 1 for i in range(width))
        for x in outputs
    )
    actual, m = certified_and_channel_map(gates, before)
    assert actual == after and m == events
    restored, back_m = certified_and_channel_map(inverse, actual)
    assert restored == before and back_m == events
    if not raw_branches:
        return
    v = torch.arange(2 * len(outputs), dtype=torch.float64)
    amps = torch.complex(v + 1, 3 * v - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << (width + 1), dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for x, y in enumerate(outputs):
        for s in (0, 1):
            state[(x << (work_width + 1)) | s] = amps[2 * x + s]
            expected[(y << (work_width + 1)) | s] = amps[2 * x + s]
    for branch in product((0, 1), repeat=events):
        actual_state = forced_branch(state, gates, branch)
        torch.testing.assert_close(
            actual_state, expected / 2 ** (events / 2), rtol=0, atol=1e-12
        )
        torch.testing.assert_close(
            forced_branch(actual_state, inverse, branch),
            state / 2**events,
            rtol=0,
            atol=1e-12,
        )
