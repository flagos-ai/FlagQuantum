"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.runtime.evolution._plan import (
    _EXPLICIT_BASIS_VERSION as _EXPLICIT_BASIS_VERSION,
)
from flagquantum.runtime.evolution._plan import (
    _PIECEWISE_VERSION as _PIECEWISE_VERSION,
)
from flagquantum.runtime.evolution._plan import (
    _PRODUCT_SPACE_VERSION as _PRODUCT_SPACE_VERSION,
)
from flagquantum.runtime.evolution._plan import (
    _SCHEDULED_FIELDS as _SCHEDULED_FIELDS,
)
from flagquantum.runtime.evolution._plan import (
    _SCHEMA as _SCHEMA,
)
from flagquantum.runtime.evolution._plan import (
    _VERSION as _VERSION,
)
from flagquantum.runtime.evolution._plan import (
    LindbladPlan as LindbladPlan,
)
from flagquantum.runtime.evolution._plan import __all__ as __all__
from flagquantum.runtime.evolution._plan import (
    _canonical_json as _canonical_json,
)
from flagquantum.runtime.evolution._plan import (
    _complex_tensor as _complex_tensor,
)
from flagquantum.runtime.evolution._plan import (
    _complex_tensor_payload as _complex_tensor_payload,
)
from flagquantum.runtime.evolution._plan import (
    _decode_request as _decode_request,
)
from flagquantum.runtime.evolution._plan import (
    _identity as _identity,
)
from flagquantum.runtime.evolution._plan import (
    _request_hamiltonian_matrix as _request_hamiltonian_matrix,
)
from flagquantum.runtime.evolution._plan import (
    _serialized_basis_labels as _serialized_basis_labels,
)
from flagquantum.runtime.evolution._plan import (
    _summary as _summary,
)
from flagquantum.runtime.evolution._plan import (
    build_lindblad_plan as build_lindblad_plan,
)

_implementation = _import_module("flagquantum.runtime.evolution._plan")
_sys.modules[__name__] = _implementation
