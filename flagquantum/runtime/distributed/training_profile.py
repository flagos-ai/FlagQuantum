"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.runtime.profiles.training import (
    FLAGOS_RANK_SUMMED_GRADIENT_DISTRIBUTIONS as FLAGOS_RANK_SUMMED_GRADIENT_DISTRIBUTIONS,
)
from flagquantum.runtime.profiles.training import (
    FLAGOS_TRAINING_CASES as FLAGOS_TRAINING_CASES,
)
from flagquantum.runtime.profiles.training import (
    FLAGOS_TRAINING_DTYPES as FLAGOS_TRAINING_DTYPES,
)
from flagquantum.runtime.profiles.training import (
    FLAGOS_TRAINING_WORLD_SIZES as FLAGOS_TRAINING_WORLD_SIZES,
)
from flagquantum.runtime.profiles.training import (
    FlagOSTrainingCase as FlagOSTrainingCase,
)
from flagquantum.runtime.profiles.training import (
    FlagOSTrainingProfile as FlagOSTrainingProfile,
)
from flagquantum.runtime.profiles.training import (
    FlagOSTrainingRun as FlagOSTrainingRun,
)
from flagquantum.runtime.profiles.training import __all__ as __all__
from flagquantum.runtime.profiles.training import (
    build_training_profile as build_training_profile,
)
from flagquantum.runtime.profiles.training import (
    training_case_from_dict as training_case_from_dict,
)
from flagquantum.runtime.profiles.training import (
    training_run_from_dict as training_run_from_dict,
)

_implementation = _import_module("flagquantum.runtime.profiles.training")
_sys.modules[__name__] = _implementation
