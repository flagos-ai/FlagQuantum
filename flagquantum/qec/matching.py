"""Historical imports for :mod:`flagquantum.qec.decoders.matching`; Core-owned compatibility boundary."""

from __future__ import annotations

import heapq as heapq
import math as math
from collections.abc import Iterable as Iterable
from dataclasses import dataclass as dataclass
from numbers import Integral as Integral
from numbers import Real as Real

from flagquantum.errors import CapabilityError as CapabilityError
from flagquantum.qec.decoders.matching import _BOUNDARY_PARTNER as _BOUNDARY_PARTNER
from flagquantum.qec.decoders.matching import (
    _DEFAULT_DEFECT_BUDGET as _DEFAULT_DEFECT_BUDGET,
)
from flagquantum.qec.decoders.matching import (
    MatchingDecodeResult as MatchingDecodeResult,
)
from flagquantum.qec.decoders.matching import (
    MinimumWeightMatchingDecoder as MinimumWeightMatchingDecoder,
)
from flagquantum.qec.decoders.matching import __all__ as __all__
from flagquantum.qec.decoders.matching import _boundary_weights as _boundary_weights
from flagquantum.qec.decoders.matching import _defects as _defects
from flagquantum.qec.decoders.matching import _detector_adjacency as _detector_adjacency
from flagquantum.qec.decoders.matching import _pair_weights as _pair_weights
from flagquantum.qec.decoders.matching import _pairing as _pairing
from flagquantum.qec.decoders.matching import _search as _search
from flagquantum.qec.decoders.matching import _SearchTree as _SearchTree
from flagquantum.qec.decoders.matching import _toggle as _toggle
from flagquantum.qec.dem.graph import DecodingGraph as DecodingGraph
from flagquantum.qec.dem.graph import DecodingGraphEdge as DecodingGraphEdge
from flagquantum.qec.dem.model import DetectorErrorModel as DetectorErrorModel
