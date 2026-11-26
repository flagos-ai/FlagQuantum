"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.qpu.result_parsing import (
    _extract_counts as _extract_counts,
)
from flagquantum.compute.remote.qpu.result_parsing import (
    _first_named as _first_named,
)
from flagquantum.compute.remote.qpu.result_parsing import (
    _flip_counts as _flip_counts,
)
from flagquantum.compute.remote.qpu.result_parsing import (
    _format_circuit_source as _format_circuit_source,
)
from flagquantum.compute.remote.qpu.result_parsing import (
    _normalize_counts as _normalize_counts,
)
from flagquantum.compute.remote.qpu.result_parsing import (
    _outcome_collection as _outcome_collection,
)
from flagquantum.compute.remote.qpu.result_parsing import (
    _outcome_counts as _outcome_counts,
)
from flagquantum.compute.remote.qpu.result_parsing import (
    _outcome_table as _outcome_table,
)
from flagquantum.compute.remote.qpu.result_parsing import (
    _probability_counts as _probability_counts,
)
from flagquantum.compute.remote.qpu.result_parsing import (
    _strip_barrier as _strip_barrier,
)
from flagquantum.compute.remote.qpu.result_parsing import (
    _tally_outcomes as _tally_outcomes,
)
from flagquantum.compute.remote.qpu.result_parsing import (
    _unwrap_result_envelope as _unwrap_result_envelope,
)

_implementation = _import_module("flagquantum.compute.remote.qpu.result_parsing")
_sys.modules[__name__] = _implementation
