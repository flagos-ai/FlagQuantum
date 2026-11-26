"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.optimizers.spsa import (
    _DEFAULT_FIRST_STEP as _DEFAULT_FIRST_STEP,
)
from flagquantum.algorithms.optimizers.spsa import (
    _STABILITY_FRACTION as _STABILITY_FRACTION,
)
from flagquantum.algorithms.optimizers.spsa import (
    SPSA_ASSUMPTIONS as SPSA_ASSUMPTIONS,
)
from flagquantum.algorithms.optimizers.spsa import (
    SPSA_LIMITATIONS as SPSA_LIMITATIONS,
)
from flagquantum.algorithms.optimizers.spsa import (
    Objective as Objective,
)
from flagquantum.algorithms.optimizers.spsa import (
    SPSAOptimizer as SPSAOptimizer,
)
from flagquantum.algorithms.optimizers.spsa import (
    SPSAResult as SPSAResult,
)
from flagquantum.algorithms.optimizers.spsa import __all__ as __all__

_implementation = _import_module("flagquantum.algorithms.optimizers.spsa")
_sys.modules[__name__] = _implementation
