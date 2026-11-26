"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.chemistry.models.orbital_rotation import (
    _real_integral as _real_integral,
)
from flagquantum.algorithms.chemistry.models.orbital_rotation import (
    closed_shell_determinant_energy as closed_shell_determinant_energy,
)
from flagquantum.algorithms.chemistry.models.orbital_rotation import (
    rotate_spatial_integrals as rotate_spatial_integrals,
)

_implementation = _import_module(
    "flagquantum.algorithms.chemistry.models.orbital_rotation"
)
_sys.modules[__name__] = _implementation
