"""Native batch costs follow upstream fixed-point scaling and buffer ownership."""

import math
import sys

import pytest
import torch

from flagquantum.errors import CapabilityError
from flagquantum.qec import DecodingGraph, PyMatchingDecoder
from flagquantum.qec.decoders.sparse_blossom import SparseBlossomDecoder
from flagquantum.qec.dem.graph import DecodingGraphEdge
from flagquantum.qec.execution.detector_sampling import _packed

pytestmark = pytest.mark.integration
pytest.importorskip("flagquantum.qec.decoders._sparse_blossom")

FORMATS = ((False, False), (False, True), (True, False), (True, True))


@pytest.mark.parametrize(("packed_input", "packed_output"), FORMATS)
@pytest.mark.parametrize(("detectors", "observables"), ((9, 1), (9, 9), (65, 64)))
@pytest.mark.parametrize(
    "costs",
    (
        (0.0,),
        (0.0, 1.0, 2.0),
        (math.log(7), math.log(9)),
        (1.0, math.log(7), 0.0),
        (1e-8, 0.0, 100.0),
    ),
)
def test_weighted_boundary_oracle_without_pymatching(
    packed_input: bool,
    packed_output: bool,
    detectors: int,
    observables: int,
    costs: tuple[float, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pymatching", None)
    monkeypatch.setattr(torch.Tensor, "numpy", None)
    edges = tuple(
        DecodingGraphEdge(
            (i, detectors),
            1 / (1 + math.exp(costs[i % len(costs)])),
            (observables - 1 - i % observables,),
        )
        for i in range(detectors)
    )
    decoder = SparseBlossomDecoder(DecodingGraph(detectors, observables, edges))

    def reject_unweighted(*args: object) -> None:
        pytest.fail("weighted decoding must produce both outputs in one native call")

    monkeypatch.setattr(decoder, "_decode", reject_unweighted)
    monkeypatch.setattr(decoder, "_decode_packed", reject_unweighted)
    dense = torch.randint(
        2, (37, detectors), generator=torch.Generator().manual_seed(41)
    )
    dense[0] = 0
    dense[1] = 1
    logicals = torch.zeros((37, observables), dtype=torch.uint8)
    for i, edge in enumerate(edges):
        logicals[:, edge.observables[0]] ^= dense[:, i].to(torch.uint8)
    expected = _packed(logicals) if packed_output else logicals.bool()

    # Independently compute the doubled fixed-point cost and restore its scale.
    original = [edge.weight for edge in edges]
    scale = (
        1.0
        if all(w.is_integer() for w in original)
        else ((1 << 24) - 1) / max(original)
    )
    integer = torch.tensor([2 * math.floor(scale * w + 0.5) for w in original])
    expected_costs = (dense @ integer).double() / (2 * scale)
    events = _packed(dense) if packed_input else dense
    storage = torch.zeros((37, events.shape[1] * 2), dtype=events.dtype)
    storage[:, ::2] = events
    strided = storage[:, ::2]
    options = dict(bit_packed_shots=packed_input, bit_packed_predictions=packed_output)
    bits, weights = decoder.decode_batch_with_weights(strided, **options)
    assert torch.equal(bits, expected)
    assert torch.equal(weights, expected_costs)
    assert weights.dtype == torch.float64 and weights.shape == (37,)
    assert bits.device.type == weights.device.type == "cpu"
    assert torch.equal(strided, events)
    bits.fill_(0)
    weights.fill_(-1)
    other_bits, other_weights = decoder.decode_batch_with_weights(strided, **options)
    assert torch.equal(other_bits, expected)
    assert torch.equal(other_weights, expected_costs)
    empty_bits, empty_weights = decoder.decode_batch_with_weights(
        strided[:0], **options
    )
    assert empty_bits.shape == (0, expected.shape[1])
    assert empty_bits.dtype == expected.dtype
    assert empty_weights.shape == (0,) and empty_weights.dtype == torch.float64


@pytest.mark.parametrize(("packed_input", "packed_output"), FORMATS)
def test_weighted_failure_recovery(packed_input: bool, packed_output: bool) -> None:
    decoder = SparseBlossomDecoder(
        DecodingGraph(9, 9, (DecodingGraphEdge((0, 9), 0.1, (8,)),))
    )
    good = torch.zeros((1, 9), dtype=torch.uint8)
    good[0, 0] = 1
    invalid = torch.cat((good, good.clone()))
    invalid[1, 1] = 1
    options = dict(bit_packed_shots=packed_input, bit_packed_predictions=packed_output)
    events = _packed(good) if packed_input else good
    for _ in range(4):
        with pytest.raises(CapabilityError, match="No perfect matching"):
            decoder.decode_batch_with_weights(
                _packed(invalid) if packed_input else invalid, **options
            )
        bits, weights = decoder.decode_batch_with_weights(events, **options)
        expected = torch.zeros((1, 9), dtype=torch.uint8)
        expected[0, 8] = 1
        assert torch.equal(
            bits, _packed(expected) if packed_output else expected.bool()
        )
        assert weights.tolist() == [pytest.approx(math.log(9))]
        assert decoder.decode_batch(good).tolist() == [
            expected.bool().flatten().tolist()
        ]


def test_weighted_input_validation() -> None:
    decoder = SparseBlossomDecoder(
        DecodingGraph(9, 1, (DecodingGraphEdge((0, 9), 0.1, (0,)),))
    )
    good = torch.zeros((2, 9), dtype=torch.uint8)
    for flag in (1, None, "true"):
        with pytest.raises(TypeError, match="flags"):
            decoder.decode_batch_with_weights(good, bit_packed_shots=flag)
        with pytest.raises(TypeError, match="flags"):
            decoder.decode_batch_with_weights(good, bit_packed_predictions=flag)
    for bad in (None, [[0]], good.float(), good.to_sparse()):
        with pytest.raises(TypeError):
            decoder.decode_batch_with_weights(bad)
    with pytest.raises(ValueError, match="binary"):
        decoder.decode_batch_with_weights(torch.full_like(good, 2))
    with pytest.raises(ValueError, match="column"):
        decoder.decode_batch_with_weights(good[:, :8])
    with pytest.raises(ValueError, match="padding"):
        decoder.decode_batch_with_weights(
            torch.tensor([[0, 2]], dtype=torch.uint8), bit_packed_shots=True
        )
    with pytest.raises(TypeError, match="uint8"):
        decoder.decode_batch_with_weights(
            torch.zeros((2, 2), dtype=torch.bool), bit_packed_shots=True
        )
    with pytest.raises(CapabilityError, match="CPU"):
        decoder.decode_batch_with_weights(
            torch.empty((2, 9), device="meta", dtype=torch.uint8)
        )


@pytest.mark.parametrize(("packed_input", "packed_output"), FORMATS)
def test_weighted_zero_cost_ties_match_upstream_costs(
    packed_input: bool, packed_output: bool
) -> None:
    pytest.importorskip("pymatching")
    graph = DecodingGraph(
        2,
        1,
        (
            DecodingGraphEdge((0, 1), 0.5, (0,)),
            DecodingGraphEdge((0, 2), 0.5),
            DecodingGraphEdge((1, 2), 0.5),
        ),
    )
    dense = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.uint8)
    batch = _packed(dense) if packed_input else dense
    options = dict(bit_packed_shots=packed_input, bit_packed_predictions=packed_output)
    bits, weights = SparseBlossomDecoder(graph).decode_batch_with_weights(
        batch, **options
    )
    _, reference = PyMatchingDecoder.from_decoding_graph(
        graph
    ).decode_batch_with_weights(batch, **options)
    assert torch.equal(weights, reference)
    assert weights.tolist() == [0.0] * 4
    assert bits.shape == (4, 1)


@pytest.mark.parametrize(("packed_input", "packed_output"), FORMATS)
def test_weighted_empty_outputs_ignore_default_device(
    packed_input: bool, packed_output: bool
) -> None:
    decoder = SparseBlossomDecoder(
        DecodingGraph(9, 9, (DecodingGraphEdge((0, 9), 0.1, (8,)),))
    )
    events = torch.empty((0, 2 if packed_input else 9), dtype=torch.uint8, device="cpu")
    with torch.device("meta"):
        bits, weights = decoder.decode_batch_with_weights(
            events,
            bit_packed_shots=packed_input,
            bit_packed_predictions=packed_output,
        )
    assert bits.device.type == weights.device.type == "cpu"
    assert bits.shape == (0, 2 if packed_output else 9)
    assert bits.dtype == (torch.uint8 if packed_output else torch.bool)
    assert weights.shape == (0,) and weights.dtype == torch.float64
