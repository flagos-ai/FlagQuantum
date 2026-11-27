"""Historical import facade for the relocated implementation."""

from importlib import import_module as _import_module

from flagquantum.algorithms.estimation.amplitude_estimation import (
    AmplitudeEstimationResult as AmplitudeEstimationResult,
)
from flagquantum.algorithms.estimation.amplitude_estimation import __all__ as __all__
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
