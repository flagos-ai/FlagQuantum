"""Historical imports for :mod:`flagquantum.qec.decoders.belief_propagation`; Core-owned compatibility boundary."""

from __future__ import annotations

import math as math
from collections.abc import Iterable as Iterable
from dataclasses import dataclass as dataclass
from numbers import Integral as Integral
from numbers import Real as Real

from flagquantum.errors import CapabilityError as CapabilityError
from flagquantum.qec.decoders.belief_propagation import _CLIP as _CLIP
from flagquantum.qec.decoders.belief_propagation import (
    _DEFAULT_MAX_ITERATIONS as _DEFAULT_MAX_ITERATIONS,
)
from flagquantum.qec.decoders.belief_propagation import _TANH_LIMIT as _TANH_LIMIT
from flagquantum.qec.decoders.belief_propagation import (
    BeliefPropagationDecoder as BeliefPropagationDecoder,
)
from flagquantum.qec.decoders.belief_propagation import (
    BeliefPropagationDecodeResult as BeliefPropagationDecodeResult,
)
from flagquantum.qec.decoders.belief_propagation import __all__ as __all__
from flagquantum.qec.decoders.belief_propagation import (
    _check_messages as _check_messages,
)
from flagquantum.qec.decoders.belief_propagation import _clip as _clip
from flagquantum.qec.decoders.belief_propagation import _decision as _decision
from flagquantum.qec.decoders.belief_propagation import _explains as _explains
from flagquantum.qec.decoders.belief_propagation import _factor_graph as _factor_graph
from flagquantum.qec.decoders.belief_propagation import _FactorGraph as _FactorGraph
from flagquantum.qec.decoders.belief_propagation import _flips as _flips
from flagquantum.qec.decoders.belief_propagation import (
    _model_from_graph as _model_from_graph,
)
from flagquantum.qec.decoders.belief_propagation import (
    _ordered_statistics as _ordered_statistics,
)
from flagquantum.qec.decoders.belief_propagation import _pivot_mask as _pivot_mask
from flagquantum.qec.decoders.belief_propagation import _posteriors as _posteriors
from flagquantum.qec.decoders.belief_propagation import _syndrome_mask as _syndrome_mask
from flagquantum.qec.decoders.belief_propagation import (
    _variable_messages as _variable_messages,
)
from flagquantum.qec.decoders.belief_propagation import _weight as _weight
from flagquantum.qec.dem.graph import DecodingGraph as DecodingGraph
from flagquantum.qec.dem.model import DemError as DemError
from flagquantum.qec.dem.model import DetectorErrorModel as DetectorErrorModel
from flagquantum.qec.dem.model import _count as _count
from flagquantum.qec.dem.model import _normalized_indices as _normalized_indices
