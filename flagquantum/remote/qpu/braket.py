"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.qpu.braket import (
    AmazonBraketProvider as AmazonBraketProvider,
)
from flagquantum.compute.remote.qpu.braket import (
    BraketSubmissionPreview as BraketSubmissionPreview,
)
from flagquantum.compute.remote.qpu.braket import (
    _braket_coupling_map as _braket_coupling_map,
)
from flagquantum.compute.remote.qpu.braket import (
    _braket_dynamic_groups as _braket_dynamic_groups,
)
from flagquantum.compute.remote.qpu.braket import (
    _first_property as _first_property,
)
from flagquantum.compute.remote.qpu.braket import (
    _property_value as _property_value,
)
from flagquantum.compute.remote.qpu.braket import (
    braket_backend_profile as braket_backend_profile,
)

_implementation = _import_module("flagquantum.compute.remote.qpu.braket")
_sys.modules[__name__] = _implementation
