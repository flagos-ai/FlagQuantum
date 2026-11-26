"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.error_mitigation.readout import (
    MAX_CORRELATED_BLOCK_QUBITS as MAX_CORRELATED_BLOCK_QUBITS,
)
from flagquantum.algorithms.error_mitigation.readout import (
    NORMALIZATION_TOLERANCE as NORMALIZATION_TOLERANCE,
)
from flagquantum.algorithms.error_mitigation.readout import (
    READOUT_MITIGATION_ASSUMPTIONS as READOUT_MITIGATION_ASSUMPTIONS,
)
from flagquantum.algorithms.error_mitigation.readout import (
    READOUT_MITIGATION_LIMITATIONS as READOUT_MITIGATION_LIMITATIONS,
)
from flagquantum.algorithms.error_mitigation.readout import (
    READOUT_MITIGATION_SCHEMA as READOUT_MITIGATION_SCHEMA,
)
from flagquantum.algorithms.error_mitigation.readout import (
    SINGULAR_VALUE_FLOOR as SINGULAR_VALUE_FLOOR,
)
from flagquantum.algorithms.error_mitigation.readout import (
    ReadoutBlock as ReadoutBlock,
)
from flagquantum.algorithms.error_mitigation.readout import (
    ReadoutMitigationPlan as ReadoutMitigationPlan,
)
from flagquantum.algorithms.error_mitigation.readout import (
    ReadoutMitigationResult as ReadoutMitigationResult,
)
from flagquantum.algorithms.error_mitigation.readout import __all__ as __all__
from flagquantum.algorithms.error_mitigation.readout import (
    _apply_block as _apply_block,
)
from flagquantum.algorithms.error_mitigation.readout import (
    _build_block as _build_block,
)
from flagquantum.algorithms.error_mitigation.readout import (
    _check_observed as _check_observed,
)
from flagquantum.algorithms.error_mitigation.readout import (
    _checked_qubits as _checked_qubits,
)
from flagquantum.algorithms.error_mitigation.readout import (
    _probability_matrix as _probability_matrix,
)
from flagquantum.algorithms.error_mitigation.readout import (
    _square_matrix as _square_matrix,
)
from flagquantum.algorithms.error_mitigation.readout import (
    plan_readout_mitigation as plan_readout_mitigation,
)
from flagquantum.algorithms.error_mitigation.readout import (
    run_readout_mitigation as run_readout_mitigation,
)
from flagquantum.algorithms.error_mitigation.readout import (
    run_readout_mitigation_counts as run_readout_mitigation_counts,
)

_implementation = _import_module("flagquantum.algorithms.error_mitigation.readout")
_sys.modules[__name__] = _implementation
