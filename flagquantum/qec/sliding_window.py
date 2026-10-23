"""Historical imports for :mod:`flagquantum.qec.decoders.sliding_window`; Core-owned compatibility boundary."""

from __future__ import annotations

import math as math
from collections.abc import Iterable as Iterable
from collections.abc import Sequence as Sequence
from dataclasses import dataclass as dataclass
from numbers import Integral as Integral
from typing import Any as Any

from flagquantum.errors import CapabilityError as CapabilityError
from flagquantum.qec.decoders.matching import (
    _DEFAULT_DEFECT_BUDGET as _DEFAULT_DEFECT_BUDGET,
)
from flagquantum.qec.decoders.matching import (
    MatchingDecodeResult as MatchingDecodeResult,
)
from flagquantum.qec.decoders.matching import (
    MinimumWeightMatchingDecoder as MinimumWeightMatchingDecoder,
)
from flagquantum.qec.decoders.matching import _defects as _defects
from flagquantum.qec.decoders.registry import get_decoder as get_decoder
from flagquantum.qec.decoders.sliding_window import (
    SlidingWindowDecoder as SlidingWindowDecoder,
)
from flagquantum.qec.decoders.sliding_window import (
    SlidingWindowDecodeResult as SlidingWindowDecodeResult,
)
from flagquantum.qec.decoders.sliding_window import (
    SlidingWindowMatchingDecoder as SlidingWindowMatchingDecoder,
)
from flagquantum.qec.decoders.sliding_window import __all__ as __all__
from flagquantum.qec.decoders.sliding_window import _mechanism_span as _mechanism_span
from flagquantum.qec.decoders.sliding_window import _ordering_key as _ordering_key
from flagquantum.qec.decoders.sliding_window import (
    _projected_columns as _projected_columns,
)
from flagquantum.qec.decoders.sliding_window import _round_sequence as _round_sequence
from flagquantum.qec.decoders.sliding_window import _rounds_of as _rounds_of
from flagquantum.qec.decoders.sliding_window import _seam_width as _seam_width
from flagquantum.qec.decoders.sliding_window import _window_columns as _window_columns
from flagquantum.qec.decoders.sliding_window import _window_graph as _window_graph
from flagquantum.qec.decoders.sliding_window import (
    dem_chunks_to_d_sparse as dem_chunks_to_d_sparse,
)
from flagquantum.qec.decoders.sliding_window import (
    dem_chunks_to_o_sparse as dem_chunks_to_o_sparse,
)
from flagquantum.qec.decoders.sliding_window import (
    dem_chunks_to_pcm as dem_chunks_to_pcm,
)
from flagquantum.qec.dem.chunks import DemChunk as DemChunk
from flagquantum.qec.dem.chunks import DemChunksSpec as DemChunksSpec
from flagquantum.qec.dem.chunks import SeamId as SeamId
from flagquantum.qec.dem.chunks import dem_chunks_from_spec as dem_chunks_from_spec
from flagquantum.qec.dem.chunks import dem_close_all as dem_close_all
from flagquantum.qec.dem.graph import DecodingGraph as DecodingGraph
from flagquantum.qec.dem.graph import DecodingGraphEdge as DecodingGraphEdge
from flagquantum.qec.dem.model import DetectorErrorModel as DetectorErrorModel
