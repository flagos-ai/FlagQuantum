"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.estimation.amplitude_estimation import (
    _LIKELIHOOD_FLOOR as _LIKELIHOOD_FLOOR,
)
from flagquantum.algorithms.estimation.amplitude_estimation import (
    AmplitudeEstimationResult as AmplitudeEstimationResult,
)
from flagquantum.algorithms.estimation.amplitude_estimation import __all__ as __all__
from flagquantum.algorithms.estimation.amplitude_estimation import (
    _amplitude_grid as _amplitude_grid,
)
from flagquantum.algorithms.estimation.amplitude_estimation import (
    _counting_probability as _counting_probability,
)
from flagquantum.algorithms.estimation.amplitude_estimation import (
    _GroverOperator as _GroverOperator,
)
from flagquantum.algorithms.estimation.amplitude_estimation import (
    _resonant_term as _resonant_term,
)
from flagquantum.algorithms.estimation.amplitude_estimation import (
    _validate_counting_width as _validate_counting_width,
)
from flagquantum.algorithms.estimation.amplitude_estimation import (
    amplitude_estimation_circuit as amplitude_estimation_circuit,
)
from flagquantum.algorithms.estimation.amplitude_estimation import (
    amplitude_resolution as amplitude_resolution,
)
from flagquantum.algorithms.estimation.amplitude_estimation import (
    maximum_likelihood_estimate as maximum_likelihood_estimate,
)
from flagquantum.algorithms.estimation.amplitude_estimation import (
    run_amplitude_estimation as run_amplitude_estimation,
)

_implementation = _import_module(
    "flagquantum.algorithms.estimation.amplitude_estimation"
)
_sys.modules[__name__] = _implementation
