"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.chemistry.molecular import (
    _SCF_MAX_ITERATIONS as _SCF_MAX_ITERATIONS,
)
from flagquantum.algorithms.chemistry.molecular import (
    _SCF_TOLERANCE as _SCF_TOLERANCE,
)
from flagquantum.algorithms.chemistry.molecular import (
    _SPIN_PATTERNS as _SPIN_PATTERNS,
)
from flagquantum.algorithms.chemistry.molecular import (
    MAX_FCI_DETERMINANTS as MAX_FCI_DETERMINANTS,
)
from flagquantum.algorithms.chemistry.molecular import (
    MOLECULAR_ASSUMPTIONS as MOLECULAR_ASSUMPTIONS,
)
from flagquantum.algorithms.chemistry.molecular import (
    MOLECULAR_LIMITATIONS as MOLECULAR_LIMITATIONS,
)
from flagquantum.algorithms.chemistry.molecular import (
    HartreeFockSolution as HartreeFockSolution,
)
from flagquantum.algorithms.chemistry.molecular import (
    MolecularHamiltonian as MolecularHamiltonian,
)
from flagquantum.algorithms.chemistry.molecular import __all__ as __all__
from flagquantum.algorithms.chemistry.molecular import (
    _determinant_masks as _determinant_masks,
)
from flagquantum.algorithms.chemistry.molecular import (
    _hartree_fock_energy as _hartree_fock_energy,
)
from flagquantum.algorithms.chemistry.molecular import (
    _sector_dimension as _sector_dimension,
)
from flagquantum.algorithms.chemistry.molecular import (
    create_molecular_hamiltonian as create_molecular_hamiltonian,
)
from flagquantum.algorithms.chemistry.molecular import (
    exact_ground_state_energy as exact_ground_state_energy,
)
from flagquantum.algorithms.chemistry.molecular import (
    fold_frozen_core as fold_frozen_core,
)
from flagquantum.algorithms.chemistry.molecular import (
    molecular_orbital_integrals as molecular_orbital_integrals,
)
from flagquantum.algorithms.chemistry.molecular import (
    restricted_hartree_fock as restricted_hartree_fock,
)
from flagquantum.algorithms.chemistry.molecular import (
    second_quantized_hamiltonian as second_quantized_hamiltonian,
)

_implementation = _import_module("flagquantum.algorithms.chemistry.molecular")
_sys.modules[__name__] = _implementation
