"""Historical imports for :mod:`flagquantum.qec.decoders.bposd`; Core-owned compatibility boundary."""

from __future__ import annotations

import math as math
from collections.abc import Iterable as Iterable
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from numbers import Integral as Integral
from numbers import Real as Real

import torch as torch

from flagquantum.errors import CapabilityError as CapabilityError
from flagquantum.qec.decoders.bposd import _DEFAULT_ITERATIONS as _DEFAULT_ITERATIONS
from flagquantum.qec.decoders.bposd import _DEFAULT_SCALING as _DEFAULT_SCALING
from flagquantum.qec.decoders.bposd import _INFINITY as _INFINITY
from flagquantum.qec.decoders.bposd import _MESSAGE_LIMIT as _MESSAGE_LIMIT
from flagquantum.qec.decoders.bposd import _MINIMUM_PROBABILITY as _MINIMUM_PROBABILITY
from flagquantum.qec.decoders.bposd import (
    BeliefPropagationOsdDecoder as BeliefPropagationOsdDecoder,
)
from flagquantum.qec.decoders.bposd import (
    BeliefPropagationOsdDecodeResult as BeliefPropagationOsdDecodeResult,
)
from flagquantum.qec.decoders.bposd import __all__ as __all__
from flagquantum.qec.decoders.bposd import _indices as _indices
from flagquantum.qec.decoders.bposd import _syndrome as _syndrome
from flagquantum.qec.dem.model import DetectorErrorModel as DetectorErrorModel
