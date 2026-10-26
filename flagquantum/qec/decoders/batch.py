"""Independent sparse shots with complete existing decoder result records."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch

from ...errors import CapabilityError
from ..adapters.pymatching import PyMatchingDecoder
from .belief_propagation import BeliefPropagationDecoder, _syndrome_mask
from .bposd import BeliefPropagationOsdDecoder, _syndrome
from .matching import MinimumWeightMatchingDecoder, _defects
from .registry import DetectorErrorModelDecoder, DetectorErrorModelDecodeResult
from .sliding_window import SlidingWindowMatchingDecoder

__all__ = ("decode_batch",)

_SupportedDecoder = (
    MinimumWeightMatchingDecoder
    | SlidingWindowMatchingDecoder
    | BeliefPropagationDecoder
    | BeliefPropagationOsdDecoder
    | PyMatchingDecoder
)
_SUPPORTED_TYPES = (
    MinimumWeightMatchingDecoder,
    SlidingWindowMatchingDecoder,
    BeliefPropagationDecoder,
    BeliefPropagationOsdDecoder,
    PyMatchingDecoder,
)


def decode_batch(
    decoder: DetectorErrorModelDecoder,
    detection_events: Iterable[Iterable[int]],
) -> tuple[DetectorErrorModelDecodeResult, ...]:
    """Decode independent sparse shots, retaining every single-shot result field.

    Materialize the finite input once and validate all rows before decoding any
    row. Storage grows with input indices and output records. Decoder-specific
    duplicate, tie, convergence and capability behavior remains unchanged.

    Args:
        decoder: One of the five certified in-tree concrete independent-shot
            decoders. Stateful streams, subclasses and plugins are refused.
        detection_events: Finite shots of sparse detector indices, not dense
            Tensor rows. Empty input returns an empty tuple for a supported type.

    Returns:
        Complete existing single-shot result records in input order. Any failure
        propagates without returning a partial successful tuple.

    Raises:
        TypeError: A carrier or detector index has an invalid type. Dense Tensors
            belong to the existing PyMatching logical-only operation.
        ValueError: A row violates its decoder's bounds or duplicate policy.
        CapabilityError: The decoder type or an algorithmic syndrome is refused.

    Examples:
        >>> from flagquantum.qec import DetectorErrorModel, get_decoder
        >>> model = DetectorErrorModel.from_stim_text(
        ...     "error(0.125) D0 L0\\ndetector D0\\nlogical_observable L0\\n"
        ... )
        >>> decoder = get_decoder("minimum_weight_matching", model)
        >>> tuple(result.observables for result in decode_batch(decoder, ((), (0,))))
        ((), (0,))
    """
    if isinstance(decoder, type) or not isinstance(decoder, DetectorErrorModelDecoder):
        raise TypeError("decoder must be a detector error model decoder instance")
    if type(decoder) not in _SUPPORTED_TYPES:
        raise CapabilityError(
            "independent-shot batching requires a certified concrete decoder; "
            "stateful streams, subclasses and plugins are not supported"
        )
    guidance = (
        "detection_events must contain sparse detector-index iterables, not Tensors; "
        "use PyMatchingDecoder.decode_batch for dense logical-only predictions"
    )
    if isinstance(detection_events, torch.Tensor):
        raise TypeError(guidance)
    rows: list[tuple[int, ...]] = []
    for row in detection_events:
        if isinstance(row, torch.Tensor):
            raise TypeError(guidance)
        rows.append(tuple(row))

    supported = cast(_SupportedDecoder, decoder)
    for row in rows:
        if isinstance(
            supported,
            (
                MinimumWeightMatchingDecoder,
                SlidingWindowMatchingDecoder,
                PyMatchingDecoder,
            ),
        ):
            _defects(row, num_detectors=supported.graph.num_detectors)
        elif isinstance(supported, BeliefPropagationDecoder):
            _syndrome_mask(row, num_detectors=supported.model.num_detectors)
        else:
            _syndrome(row, num_detectors=supported.model.num_detectors)
    return tuple(decoder.decode(row) for row in rows)
