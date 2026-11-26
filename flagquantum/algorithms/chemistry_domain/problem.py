"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.chemistry.models.problem import (
    MolecularProblem as MolecularProblem,
)
from flagquantum.algorithms.chemistry.models.problem import (
    fermion_hamiltonian as fermion_hamiltonian,
)
from flagquantum.algorithms.chemistry.models.problem import (
    from_native_molecule as from_native_molecule,
)

_implementation = _import_module("flagquantum.algorithms.chemistry.models.problem")
_sys.modules[__name__] = _implementation
