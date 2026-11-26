"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.runtime.profiles.capacity import (
    CAPACITY_MODES as CAPACITY_MODES,
)
from flagquantum.runtime.profiles.capacity import (
    CAPACITY_PROFILE_SCHEMA as CAPACITY_PROFILE_SCHEMA,
)
from flagquantum.runtime.profiles.capacity import (
    CAPACITY_RUN_SCHEMA as CAPACITY_RUN_SCHEMA,
)
from flagquantum.runtime.profiles.capacity import (
    FlagOSCapacityProfileError as FlagOSCapacityProfileError,
)
from flagquantum.runtime.profiles.capacity import (
    FlagOSStatevectorCapacityProfile as FlagOSStatevectorCapacityProfile,
)
from flagquantum.runtime.profiles.capacity import __all__ as __all__
from flagquantum.runtime.profiles.capacity import (
    _oom_accepted as _oom_accepted,
)
from flagquantum.runtime.profiles.capacity import (
    _require as _require,
)
from flagquantum.runtime.profiles.capacity import (
    _sharded_accepted as _sharded_accepted,
)
from flagquantum.runtime.profiles.capacity import (
    _validate_boundary as _validate_boundary,
)
from flagquantum.runtime.profiles.capacity import (
    _validate_topology as _validate_topology,
)
from flagquantum.runtime.profiles.capacity import (
    build_flagos_statevector_capacity_profile as build_flagos_statevector_capacity_profile,
)
from flagquantum.runtime.profiles.capacity import (
    json_key as json_key,
)

_implementation = _import_module("flagquantum.runtime.profiles.capacity")
_sys.modules[__name__] = _implementation
