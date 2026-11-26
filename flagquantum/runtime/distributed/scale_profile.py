"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.runtime.profiles.scale import (
    FLAGOS_SCALE_CASES as FLAGOS_SCALE_CASES,
)
from flagquantum.runtime.profiles.scale import (
    FLAGOS_SCALE_DTYPES as FLAGOS_SCALE_DTYPES,
)
from flagquantum.runtime.profiles.scale import (
    FLAGOS_SCALE_WORLD_SIZES as FLAGOS_SCALE_WORLD_SIZES,
)
from flagquantum.runtime.profiles.scale import (
    FlagOSStatevectorScaleCase as FlagOSStatevectorScaleCase,
)
from flagquantum.runtime.profiles.scale import (
    FlagOSStatevectorScaleProfile as FlagOSStatevectorScaleProfile,
)
from flagquantum.runtime.profiles.scale import (
    FlagOSStatevectorScaleRun as FlagOSStatevectorScaleRun,
)
from flagquantum.runtime.profiles.scale import __all__ as __all__
from flagquantum.runtime.profiles.scale import (
    build_scale_profile as build_scale_profile,
)
from flagquantum.runtime.profiles.scale import (
    scale_case_from_dict as scale_case_from_dict,
)
from flagquantum.runtime.profiles.scale import (
    scale_run_from_dict as scale_run_from_dict,
)

_implementation = _import_module("flagquantum.runtime.profiles.scale")
_sys.modules[__name__] = _implementation
