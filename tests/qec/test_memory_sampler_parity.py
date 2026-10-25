"""Parity extraction preserves GF(2) records without per-column tensor launches."""

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from flagquantum.qec.execution.sampling import _parity_bits

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("shots", [0, 1, 9])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.int64])
def test_ragged_parities_match_independent_binary_reference(
    shots: int, dtype: torch.dtype
) -> None:
    generator = torch.Generator().manual_seed(53)
    record = torch.randint(2, (shots, 162), generator=generator).to(dtype)[:, ::2]
    original = record.clone()
    columns = ((), (0, 0), (1,), (80, 0, 17, 17), tuple(range(81)), (80, 80, 80))
    expected = torch.empty((shots, len(columns)), dtype=torch.int8)
    for row in range(shots):
        for index, parity in enumerate(columns):
            expected[row, index] = (
                sum(int(record[row, column]) for column in parity) % 2
            )
    actual = _parity_bits(columns, record)
    assert actual.dtype == torch.int8 and actual.device.type == "cpu"
    assert torch.equal(actual, expected)
    actual.fill_(1)
    assert torch.equal(record, original)
    assert _parity_bits((), record).shape == (shots, 0)


@pytest.mark.parametrize("width", [255, 256, 257, 513])
def test_long_products_preserve_parity_across_byte_wrap(width: int) -> None:
    record = torch.ones((2, width), dtype=torch.int64)
    record[1, 0] = 0
    result = _parity_bits((tuple(range(width)),), record)
    assert result.tolist() == [[width % 2], [(width - 1) % 2]]


class _TensorCalls(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __torch_dispatch__(
        self,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        self.calls += 1
        return func(*args, **(kwargs or {}))


def test_equal_width_parities_do_not_launch_tensor_work_per_output() -> None:
    record = torch.arange(8 * 130).reshape(8, 130) % 2
    columns = tuple((index, index + 1, index + 2) for index in range(128))
    counter = _TensorCalls()
    with counter:
        result = _parity_bits(columns, record)
    assert torch.equal(result, record[:, 1:129].to(torch.int8))
    assert counter.calls <= 64, f"per-output tensor launch overhead: {counter.calls}"
