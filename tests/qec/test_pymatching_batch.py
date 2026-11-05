"""Approved batch logical prediction contract and independent SDK conformance."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("pymatching")
import pymatching

from flagquantum.errors import CapabilityError
from flagquantum.qec import DemError, DetectorErrorModel, PyMatchingDecoder

pytestmark = pytest.mark.integration


@pytest.fixture
def decoder() -> PyMatchingDecoder:
    return PyMatchingDecoder.from_detector_error_model(
        DetectorErrorModel(
            2,
            2,
            (DemError(0.1, (0,), (0,)), DemError(0.2, (1,), (1,))),
        )
    )


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ],
)
def test_all_binary_integer_types_match_independent_upstream(
    decoder: PyMatchingDecoder, dtype: torch.dtype
) -> None:
    events = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=dtype)
    original = events.clone()
    reference = pymatching.Matching()
    reference.add_boundary_edge(0, fault_ids={0}, weight=2.1972245773362196)
    reference.add_boundary_edge(1, fault_ids={1}, weight=1.3862943611198906)
    expected = reference.decode_batch(events.to(torch.uint8).numpy())
    result = decoder.decode_batch(events)
    assert result.dtype == torch.bool
    assert result.device.type == "cpu"
    assert torch.equal(result, torch.as_tensor(expected, dtype=torch.bool))
    assert torch.equal(events, original)


def test_noncontiguous_input_and_single_row(decoder: PyMatchingDecoder) -> None:
    events = torch.tensor([[1, 9, 0, 9], [0, 9, 1, 9]])[:, ::2]
    assert not events.is_contiguous()
    assert decoder.decode_batch(events).tolist() == [[True, False], [False, True]]
    assert decoder.decode_batch(events[:1]).shape == (1, 2)


def test_empty_batch_does_not_call_sdk(decoder: PyMatchingDecoder) -> None:
    sdk = Mock()
    object.__setattr__(decoder, "_matching", sdk)
    assert decoder.decode_batch(torch.empty((0, 2), dtype=torch.bool)).shape == (0, 2)
    sdk.decode_batch.assert_not_called()


def test_no_observables_still_validates_reachability() -> None:
    decoder = PyMatchingDecoder.from_detector_error_model(
        DetectorErrorModel(2, 0, (DemError(0.1, (0, 1)),))
    )
    assert decoder.decode_batch(torch.tensor([[0, 0], [1, 1]])).shape == (2, 0)
    with pytest.raises(CapabilityError, match="could not decode the batch"):
        decoder.decode_batch(torch.tensor([[0, 0], [1, 0]]))


@pytest.mark.parametrize(
    ("events", "error", "message"),
    [
        ([[0, 0]], TypeError, "torch.Tensor"),
        (torch.zeros((1, 2)), TypeError, "dtype"),
        (torch.zeros((1, 2), dtype=torch.complex64), TypeError, "dtype"),
        (torch.zeros((1, 2), dtype=torch.int64).to_sparse(), TypeError, "dense"),
        (torch.zeros((1, 2), dtype=torch.bool, device="meta"), CapabilityError, "CPU"),
        (torch.zeros(2, dtype=torch.bool), ValueError, "shape"),
        (torch.zeros((1, 1, 2), dtype=torch.bool), ValueError, "shape"),
        (torch.zeros((0, 3), dtype=torch.bool), ValueError, "columns"),
        (torch.tensor([[0, -1]]), ValueError, "binary"),
        (torch.tensor([[0, 256]]), ValueError, "binary"),
    ],
)
def test_invalid_input_fails_before_sdk(
    decoder: PyMatchingDecoder, events: Any, error: type[Exception], message: str
) -> None:
    # Pytest deliberately supplies invalid carriers at this runtime API boundary.
    sdk = Mock()
    object.__setattr__(decoder, "_matching", sdk)
    with pytest.raises(error, match=message):
        decoder.decode_batch(events)
    sdk.decode_batch.assert_not_called()


def test_one_sdk_call_and_buffer_ownership(decoder: PyMatchingDecoder) -> None:
    events = torch.tensor([[1, 0]], dtype=torch.uint8)
    answer = torch.tensor([[1, 0]], dtype=torch.uint8).numpy()

    def decode_batch(buffer: Any) -> object:
        # The external SDK owns this mutable NumPy buffer, not a domain carrier.
        buffer[:] = 0
        return answer

    sdk = SimpleNamespace(decode_batch=Mock(side_effect=decode_batch))
    object.__setattr__(decoder, "_matching", sdk)
    result = decoder.decode_batch(events)
    assert sdk.decode_batch.call_count == 1
    assert decoder._matching is sdk
    assert events.tolist() == [[1, 0]]
    answer[:] = 0
    assert result.tolist() == [[True, False]]


@pytest.mark.parametrize("answer", [torch.zeros((1, 3)), torch.tensor([[0, 2]])])
def test_sdk_output_is_validated(
    decoder: PyMatchingDecoder, answer: torch.Tensor
) -> None:
    sdk = SimpleNamespace(decode_batch=Mock(return_value=answer.numpy()))
    object.__setattr__(decoder, "_matching", sdk)
    with pytest.raises(CapabilityError, match="returned"):
        decoder.decode_batch(torch.tensor([[0, 0]]))


@pytest.mark.parametrize("width", [3, 8, 9, 16, 17])
def test_packed_independent_boundary_oracle(width: int) -> None:
    model = DetectorErrorModel(
        width, width, tuple(DemError(0.1, (i,), (width - i - 1,)) for i in range(width))
    )
    decoder = PyMatchingDecoder.from_detector_error_model(model)
    # Every syndrome has a unique boundary correction; reverse its logical bits.
    dense = torch.tensor(
        [[(value >> i) & 1 for i in range(width)] for value in range(8)],
        dtype=torch.uint8,
    )

    def pack(bits: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(
            (bits.shape[0], (bits.shape[1] + 7) // 8), dtype=torch.uint8
        )
        for column in range(bits.shape[1]):
            result[:, column // 8] |= bits[:, column] << (column % 8)
        return result

    packed = pack(dense)
    expected = pack(dense.flip(1))
    result = decoder.decode_packed_batch(packed)
    assert torch.equal(result, expected)
    assert torch.equal(pack(decoder.decode_batch(dense).to(torch.uint8)), expected)
    reference = pymatching.Matching()
    for i in range(width):
        reference.add_boundary_edge(i, fault_ids=width - i - 1, weight=1.0)
    direct = reference.decode_batch(
        packed.numpy(), bit_packed_shots=True, bit_packed_predictions=True
    )
    assert torch.equal(result, torch.as_tensor(direct))
    direct.fill(0)
    assert torch.equal(result, expected)


def test_packed_sdk_numpy_views_are_owned(
    monkeypatch: pytest.MonkeyPatch, decoder: PyMatchingDecoder
) -> None:
    events = torch.tensor([[1], [2]], dtype=torch.uint8)
    before = events.clone()
    sdk = Mock()

    def predict(buffer: torch.Tensor, **options: object) -> object:
        array = torch.as_tensor(buffer)
        array.zero_()
        return array.numpy()

    sdk.decode_batch.side_effect = predict
    object.__setattr__(decoder, "_matching", sdk)
    result = decoder.decode_packed_batch(events)
    assert torch.equal(events, before)
    assert result.tolist() == [[0], [0]]
    torch.as_tensor(sdk.decode_batch.call_args.args[0]).fill_(3)
    assert result.tolist() == [[0], [0]]


def test_packed_zero_observables_still_decodes() -> None:
    decoder = PyMatchingDecoder.from_detector_error_model(
        DetectorErrorModel(1, 0, (DemError(0.1, (0,), ()),))
    )
    result = decoder.decode_packed_batch(torch.tensor([[0], [1]], dtype=torch.uint8))
    assert result.shape == (2, 0)
    assert result.dtype == torch.uint8
