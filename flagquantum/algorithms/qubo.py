"""Historical import facade for the relocated implementation."""

from flagquantum.algorithms.optimization.qubo import (
    QuboProblem as QuboProblem,
)
from flagquantum.algorithms.optimization.qubo import __all__ as __all__
from flagquantum.algorithms.optimization.qubo import (
    ising_to_qubo as ising_to_qubo,
)
from flagquantum.algorithms.optimization.qubo import (
    max_cut_qubo as max_cut_qubo,
)
from flagquantum.algorithms.optimization.qubo import (
    qubo_energy as qubo_energy,
)
from flagquantum.algorithms.optimization.qubo import (
    qubo_to_ising as qubo_to_ising,
)
