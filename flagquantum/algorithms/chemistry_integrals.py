"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.chemistry.molecular.integrals import (
    _ERI_SYMMETRY as _ERI_SYMMETRY,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _P_ANGULAR as _P_ANGULAR,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _S_ANGULAR as _S_ANGULAR,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _STO3G as _STO3G,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    ATOMIC_NUMBERS as ATOMIC_NUMBERS,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    BASIS_SETS as BASIS_SETS,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    BOHR_RADIUS as BOHR_RADIUS,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    AtomicOrbital as AtomicOrbital,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    MolecularGeometry as MolecularGeometry,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    MolecularIntegrals as MolecularIntegrals,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    Vector as Vector,
)
from flagquantum.algorithms.chemistry.molecular.integrals import __all__ as __all__
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _atomic_orbitals as _atomic_orbitals,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _boys as _boys,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _differentiate as _differentiate,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _dot as _dot,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _double_factorial as _double_factorial,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _electron_repulsion as _electron_repulsion,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _evaluate as _evaluate,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _hermite_e as _hermite_e,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _kinetic as _kinetic,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _norm as _norm,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _nuclear as _nuclear,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _overlap as _overlap,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _Pair as _Pair,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _pair_primitives as _pair_primitives,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _potential_grid as _potential_grid,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _primitive_norm as _primitive_norm,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _sub as _sub,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    _triple as _triple,
)
from flagquantum.algorithms.chemistry.molecular.integrals import (
    molecular_integrals as molecular_integrals,
)

_implementation = _import_module("flagquantum.algorithms.chemistry.molecular.integrals")
_sys.modules[__name__] = _implementation
