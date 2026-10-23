"""Historical imports for :mod:`flagquantum.qec.dem.graph`; Core-owned compatibility boundary."""

from __future__ import annotations

import math as math
from dataclasses import dataclass as dataclass
from numbers import Integral as Integral

from flagquantum.errors import CapabilityError as CapabilityError
from flagquantum.qec.dem.graph import _GRAPH_DETECTORS as _GRAPH_DETECTORS
from flagquantum.qec.dem.graph import DecodingGraph as DecodingGraph
from flagquantum.qec.dem.graph import DecodingGraphEdge as DecodingGraphEdge
from flagquantum.qec.dem.graph import __all__ as __all__
from flagquantum.qec.dem.model import DetectorErrorModel as DetectorErrorModel
from flagquantum.qec.dem.model import _count as _count
from flagquantum.qec.dem.model import _normalized_indices as _normalized_indices
from flagquantum.qec.dem.model import _probability as _probability
