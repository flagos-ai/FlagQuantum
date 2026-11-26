"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.error_mitigation.folding import (
    FOLD_NOISE_SCHEMA as FOLD_NOISE_SCHEMA,
)
from flagquantum.algorithms.error_mitigation.folding import (
    FOLDING_ASSUMPTIONS as FOLDING_ASSUMPTIONS,
)
from flagquantum.algorithms.error_mitigation.folding import (
    FOLDING_PLACEMENTS as FOLDING_PLACEMENTS,
)
from flagquantum.algorithms.error_mitigation.folding import (
    FOLDING_SCHEMA as FOLDING_SCHEMA,
)
from flagquantum.algorithms.error_mitigation.folding import (
    FOLDING_STRATEGIES as FOLDING_STRATEGIES,
)
from flagquantum.algorithms.error_mitigation.folding import (
    FoldingPlacement as FoldingPlacement,
)
from flagquantum.algorithms.error_mitigation.folding import (
    FoldingPlan as FoldingPlan,
)
from flagquantum.algorithms.error_mitigation.folding import (
    FoldingStrategy as FoldingStrategy,
)
from flagquantum.algorithms.error_mitigation.folding import (
    FoldNoiseAmplification as FoldNoiseAmplification,
)
from flagquantum.algorithms.error_mitigation.folding import __all__ as __all__
from flagquantum.algorithms.error_mitigation.folding import (
    _channel_locations as _channel_locations,
)
from flagquantum.algorithms.error_mitigation.folding import (
    _noise_identity as _noise_identity,
)
from flagquantum.algorithms.error_mitigation.folding import (
    _pair_count as _pair_count,
)
from flagquantum.algorithms.error_mitigation.folding import (
    _placement_order as _placement_order,
)
from flagquantum.algorithms.error_mitigation.folding import (
    _plan_identity as _plan_identity,
)
from flagquantum.algorithms.error_mitigation.folding import (
    _require_scale_factor as _require_scale_factor,
)
from flagquantum.algorithms.error_mitigation.folding import (
    _require_unitary as _require_unitary,
)
from flagquantum.algorithms.error_mitigation.folding import (
    fold_noise_amplification as fold_noise_amplification,
)
from flagquantum.algorithms.error_mitigation.folding import (
    fold_program as fold_program,
)

_implementation = _import_module("flagquantum.algorithms.error_mitigation.folding")
_sys.modules[__name__] = _implementation
