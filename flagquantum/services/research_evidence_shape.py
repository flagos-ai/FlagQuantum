"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.optimization._research_evidence_shape import (
    _BUNDLE_FIELDS as _BUNDLE_FIELDS,
)
from flagquantum.algorithms.optimization._research_evidence_shape import (
    _BUNDLE_SCHEMA as _BUNDLE_SCHEMA,
)
from flagquantum.algorithms.optimization._research_evidence_shape import (
    _BUNDLE_STATUSES as _BUNDLE_STATUSES,
)
from flagquantum.algorithms.optimization._research_evidence_shape import (
    _CAPABILITY_FIELDS as _CAPABILITY_FIELDS,
)
from flagquantum.algorithms.optimization._research_evidence_shape import (
    _PROVENANCE_FIELDS as _PROVENANCE_FIELDS,
)
from flagquantum.algorithms.optimization._research_evidence_shape import (
    _evidence_shape_error as _evidence_shape_error,
)

_implementation = _import_module(
    "flagquantum.algorithms.optimization._research_evidence_shape"
)
_sys.modules[__name__] = _implementation
