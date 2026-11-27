"""Historical numerical import facade."""

from flagquantum.runtime.results.trajectories import (
    MPSMonteCarloResult as MPSMonteCarloResult,
)
from flagquantum.runtime.results.trajectories import (
    TrajectoryFailure as TrajectoryFailure,
)
from flagquantum.runtime.results.trajectories import (
    TrajectoryStatistics as TrajectoryStatistics,
)

__all__ = ("MPSMonteCarloResult", "TrajectoryFailure", "TrajectoryStatistics")
