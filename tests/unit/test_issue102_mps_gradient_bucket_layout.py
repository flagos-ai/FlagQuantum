import pytest
import torch

from flagquantum.runtime.backends.mps.reverse import _gradient_bucket_layout

pytestmark = pytest.mark.unit


def test_thousand_parameters_collapse_to_owner_dtype_buckets():
    parameters = tuple(torch.empty((), dtype=torch.float32) for _ in range(1000))
    owners = tuple(index % 8 for index in range(1000))
    buckets = _gradient_bucket_layout(parameters, owners, max_bucket_bytes=4096)
    assert len(buckets) == 8
    assert {owner for _, owner, _ in buckets} == set(range(8))
    assert sum(len(pieces) for _, _, pieces in buckets) == 1000


def test_bucket_layout_splits_large_parameters_and_separates_dtype():
    parameters = (
        torch.empty(9, dtype=torch.float32),
        torch.empty(3, dtype=torch.float64),
    )
    buckets = _gradient_bucket_layout(parameters, (0, 0), max_bucket_bytes=16)
    assert all(
        sum(
            (end - start) * torch.empty((), dtype=dtype).element_size()
            for _, start, end in pieces
        )
        <= 16
        for dtype, _, pieces in buckets
    )
    assert {dtype for dtype, _, _ in buckets} == {torch.float32, torch.float64}
    with pytest.raises(ValueError, match="match"):
        _gradient_bucket_layout(parameters, (0,), max_bucket_bytes=16)
    with pytest.raises(ValueError, match="positive"):
        _gradient_bucket_layout(parameters, (0, 0), max_bucket_bytes=0)
