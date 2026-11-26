"""Historical numerical import facade."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.runtime.results.trajectories import (
    MPSMonteCarloResult as MPSMonteCarloResult,
)
from flagquantum.runtime.results.trajectories import (
    TrajectoryFailure as TrajectoryFailure,
)
from flagquantum.runtime.results.trajectories import (
    TrajectoryStatistics as TrajectoryStatistics,
)

_sys.modules[__name__] = _import_module("flagquantum.runtime.results.trajectories")
