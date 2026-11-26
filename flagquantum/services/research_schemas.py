"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.optimization._research_schemas import (
    _COMPLEX_AMPLITUDE as _COMPLEX_AMPLITUDE,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _DIALECT as _DIALECT,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _EVIDENCE as _EVIDENCE,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _EXACT_REQUEST as _EXACT_REQUEST,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _EXACT_RESULT as _EXACT_RESULT,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _EXECUTION as _EXECUTION,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _ISSUE as _ISSUE,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _MANIFEST as _MANIFEST,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _PAULI_SUM as _PAULI_SUM,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _PAULI_TERM as _PAULI_TERM,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _PROVIDER_PROVENANCE as _PROVIDER_PROVENANCE,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _SCHEMAS as _SCHEMAS,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _SCHEMAS_BY_ID as _SCHEMAS_BY_ID,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _STATEVECTOR as _STATEVECTOR,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _STRUCTURED_ERROR as _STRUCTURED_ERROR,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _UNSUPPORTED as _UNSUPPORTED,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _VALIDATION_REPORT as _VALIDATION_REPORT,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _VARIANCE_REQUEST as _VARIANCE_REQUEST,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _VARIANCE_RESULT as _VARIANCE_RESULT,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _VQE_REQUEST as _VQE_REQUEST,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _VQE_RESULT as _VQE_RESULT,
)
from flagquantum.algorithms.optimization._research_schemas import (
    ResearchSchemaDocument as ResearchSchemaDocument,
)
from flagquantum.algorithms.optimization._research_schemas import __all__ as __all__
from flagquantum.algorithms.optimization._research_schemas import (
    _closed as _closed,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _document as _document,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _entry as _entry,
)
from flagquantum.algorithms.optimization._research_schemas import (
    _nullable as _nullable,
)
from flagquantum.algorithms.optimization._research_schemas import (
    describe_research_schema as describe_research_schema,
)
from flagquantum.algorithms.optimization._research_schemas import (
    list_research_schemas as list_research_schemas,
)

_implementation = _import_module(
    "flagquantum.algorithms.optimization._research_schemas"
)
_sys.modules[__name__] = _implementation
