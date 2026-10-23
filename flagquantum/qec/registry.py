"""Historical imports for :mod:`flagquantum.qec.decoders.registry`; Core-owned compatibility boundary."""

from __future__ import annotations

import inspect as inspect
from collections.abc import Callable as Callable
from collections.abc import Iterable as Iterable
from typing import Any as Any
from typing import Protocol as Protocol
from typing import TypeVar as TypeVar
from typing import runtime_checkable as runtime_checkable

from flagquantum.errors import CapabilityError as CapabilityError
from flagquantum.qec.adapters.pymatching import PyMatchingDecoder as PyMatchingDecoder
from flagquantum.qec.decoders.belief_propagation import (
    BeliefPropagationDecoder as BeliefPropagationDecoder,
)
from flagquantum.qec.decoders.bposd import (
    BeliefPropagationOsdDecoder as BeliefPropagationOsdDecoder,
)
from flagquantum.qec.decoders.matching import (
    MinimumWeightMatchingDecoder as MinimumWeightMatchingDecoder,
)
from flagquantum.qec.decoders.registry import _DECODERS as _DECODERS
from flagquantum.qec.decoders.registry import AUTHORITY_NAME as AUTHORITY_NAME
from flagquantum.qec.decoders.registry import (
    BELIEF_PROPAGATION_NAME as BELIEF_PROPAGATION_NAME,
)
from flagquantum.qec.decoders.registry import (
    BELIEF_PROPAGATION_OSD_NAME as BELIEF_PROPAGATION_OSD_NAME,
)
from flagquantum.qec.decoders.registry import CROSS_CHECK_NAME as CROSS_CHECK_NAME
from flagquantum.qec.decoders.registry import SLIDING_WINDOW_NAME as SLIDING_WINDOW_NAME
from flagquantum.qec.decoders.registry import (
    DetectorErrorModelDecoder as DetectorErrorModelDecoder,
)
from flagquantum.qec.decoders.registry import (
    DetectorErrorModelDecodeResult as DetectorErrorModelDecodeResult,
)
from flagquantum.qec.decoders.registry import (
    GraphlikeDetectorErrorModelDecoder as GraphlikeDetectorErrorModelDecoder,
)
from flagquantum.qec.decoders.registry import __annotations__ as __annotations__
from flagquantum.qec.decoders.registry import (
    _declares_a_graph_carrier as _declares_a_graph_carrier,
)
from flagquantum.qec.decoders.registry import _DecoderT as _DecoderT
from flagquantum.qec.decoders.registry import (
    _register_windowed_decoder as _register_windowed_decoder,
)
from flagquantum.qec.decoders.registry import _takes_a_graph as _takes_a_graph
from flagquantum.qec.decoders.registry import decoder_names as decoder_names
from flagquantum.qec.decoders.registry import get_decoder as get_decoder
from flagquantum.qec.decoders.registry import register_decoder as register_decoder
from flagquantum.qec.dem.graph import DecodingGraph as DecodingGraph
from flagquantum.qec.dem.model import DetectorErrorModel as DetectorErrorModel
