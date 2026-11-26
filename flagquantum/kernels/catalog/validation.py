"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.catalog.validation import (
    _ARTIFACT_REFERENCE as _ARTIFACT_REFERENCE,
)
from flagquantum.simulation.kernels.catalog.validation import (
    _BANNED_SEMANTIC_COMPONENTS as _BANNED_SEMANTIC_COMPONENTS,
)
from flagquantum.simulation.kernels.catalog.validation import (
    _CATALOG_ID as _CATALOG_ID,
)
from flagquantum.simulation.kernels.catalog.validation import (
    _COMPONENT as _COMPONENT,
)
from flagquantum.simulation.kernels.catalog.validation import (
    _DOMAIN_CODES as _DOMAIN_CODES,
)
from flagquantum.simulation.kernels.catalog.validation import (
    _EVIDENCE_ID as _EVIDENCE_ID,
)
from flagquantum.simulation.kernels.catalog.validation import (
    _IMPLEMENTATION_ID as _IMPLEMENTATION_ID,
)
from flagquantum.simulation.kernels.catalog.validation import (
    _TEST_REFERENCE as _TEST_REFERENCE,
)
from flagquantum.simulation.kernels.catalog.validation import __all__ as __all__
from flagquantum.simulation.kernels.catalog.validation import (
    _require_components as _require_components,
)
from flagquantum.simulation.kernels.catalog.validation import (
    _require_devices as _require_devices,
)
from flagquantum.simulation.kernels.catalog.validation import (
    _require_unique as _require_unique,
)
from flagquantum.simulation.kernels.catalog.validation import (
    validate_catalog as validate_catalog,
)

_implementation = _import_module("flagquantum.simulation.kernels.catalog.validation")
_sys.modules[__name__] = _implementation
