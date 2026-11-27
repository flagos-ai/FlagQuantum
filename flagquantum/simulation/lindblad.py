"""Historical import facade for the relocated implementation."""

from flagquantum.simulation.dynamics.lindblad import (
    CollapseOperator as CollapseOperator,
)
from flagquantum.simulation.dynamics.lindblad import (
    EvolutionPlan as EvolutionPlan,
)
from flagquantum.simulation.dynamics.lindblad import (
    EvolutionResult as EvolutionResult,
)
from flagquantum.simulation.dynamics.lindblad import (
    EvolutionValidationError as EvolutionValidationError,
)
from flagquantum.simulation.dynamics.lindblad import __all__ as __all__
from flagquantum.simulation.dynamics.lindblad import (
    amplitude_damping as amplitude_damping,
)
from flagquantum.simulation.dynamics.lindblad import (
    evolve_density_matrix as evolve_density_matrix,
)
from flagquantum.simulation.dynamics.lindblad import (
    plan_density_matrix_evolution as plan_density_matrix_evolution,
)
