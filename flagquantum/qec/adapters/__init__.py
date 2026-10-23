"""Historical imports for :mod:`flagquantum.qec.adapters.pymatching`; Core-owned compatibility boundary."""

from __future__ import annotations

from collections.abc import Iterable as Iterable
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from importlib import import_module as import_module
from types import ModuleType as ModuleType
from typing import Any as Any

from flagquantum.errors import CapabilityError as CapabilityError
from flagquantum.qec.adapters.pymatching import _BOUNDARY as _BOUNDARY
from flagquantum.qec.adapters.pymatching import _EXTRA_NAME as _EXTRA_NAME
from flagquantum.qec.adapters.pymatching import (
    MatchingDependencyError as MatchingDependencyError,
)
from flagquantum.qec.adapters.pymatching import PyMatchingDecoder as PyMatchingDecoder
from flagquantum.qec.adapters.pymatching import __all__ as __all__
from flagquantum.qec.adapters.pymatching import _pymatching as _pymatching
from flagquantum.qec.adapters.pymatching import _translate as _translate
from flagquantum.qec.decoders.matching import (
    MatchingDecodeResult as MatchingDecodeResult,
)
from flagquantum.qec.decoders.matching import _defects as _defects
from flagquantum.qec.dem.graph import DecodingGraph as DecodingGraph
from flagquantum.qec.dem.graph import DecodingGraphEdge as DecodingGraphEdge
from flagquantum.qec.dem.model import DetectorErrorModel as DetectorErrorModel
