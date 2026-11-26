"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.error_mitigation.cdr import (
    _MAXIMUM_TRAINING_POINTS as _MAXIMUM_TRAINING_POINTS,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    _MINIMUM_TRAINING_POINTS as _MINIMUM_TRAINING_POINTS,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    _QUARTER_TURN as _QUARTER_TURN,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    _QUARTER_TURN_WORDS as _QUARTER_TURN_WORDS,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    _SEPARATION_TOLERANCE as _SEPARATION_TOLERANCE,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    CDR_ASSUMPTIONS as CDR_ASSUMPTIONS,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    CDR_LIMITATIONS as CDR_LIMITATIONS,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    CDR_SNAP_OPCODES as CDR_SNAP_OPCODES,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    CdrResult as CdrResult,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    CliffordFit as CliffordFit,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    CliffordTrainingPoint as CliffordTrainingPoint,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    CliffordVariant as CliffordVariant,
)
from flagquantum.algorithms.error_mitigation.cdr import __all__ as __all__
from flagquantum.algorithms.error_mitigation.cdr import (
    _angle_of as _angle_of,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    _as_variant_irs as _as_variant_irs,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    _expectation as _expectation,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    _finite_floats as _finite_floats,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    _fit as _fit,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    _lowered as _lowered,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    _plan_sites as _plan_sites,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    _require_no_readout as _require_no_readout,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    _require_no_request as _require_no_request,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    _require_rewritten_names_survive as _require_rewritten_names_survive,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    _Site as _Site,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    _snapped as _snapped,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    clifford_variants as clifford_variants,
)
from flagquantum.algorithms.error_mitigation.cdr import (
    run_cdr as run_cdr,
)

_implementation = _import_module("flagquantum.algorithms.error_mitigation.cdr")
_sys.modules[__name__] = _implementation
