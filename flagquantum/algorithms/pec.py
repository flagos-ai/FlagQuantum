"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.error_mitigation.pec import (
    _DECOMPOSITION_TOLERANCE as _DECOMPOSITION_TOLERANCE,
)
from flagquantum.algorithms.error_mitigation.pec import (
    _MAXIMUM_TERMS as _MAXIMUM_TERMS,
)
from flagquantum.algorithms.error_mitigation.pec import (
    PEC_ASSUMPTIONS as PEC_ASSUMPTIONS,
)
from flagquantum.algorithms.error_mitigation.pec import (
    PEC_LIMITATIONS as PEC_LIMITATIONS,
)
from flagquantum.algorithms.error_mitigation.pec import (
    PauliTwirlDecomposition as PauliTwirlDecomposition,
)
from flagquantum.algorithms.error_mitigation.pec import (
    PecLocation as PecLocation,
)
from flagquantum.algorithms.error_mitigation.pec import (
    PecResult as PecResult,
)
from flagquantum.algorithms.error_mitigation.pec import __all__ as __all__
from flagquantum.algorithms.error_mitigation.pec import (
    _commutation_parity as _commutation_parity,
)
from flagquantum.algorithms.error_mitigation.pec import (
    _corrected as _corrected,
)
from flagquantum.algorithms.error_mitigation.pec import (
    _declared_channel as _declared_channel,
)
from flagquantum.algorithms.error_mitigation.pec import (
    _quasi_probabilities as _quasi_probabilities,
)
from flagquantum.algorithms.error_mitigation.pec import (
    _read as _read,
)
from flagquantum.algorithms.error_mitigation.pec import (
    _word_operators as _word_operators,
)
from flagquantum.algorithms.error_mitigation.pec import (
    pauli_twirl_decomposition as pauli_twirl_decomposition,
)
from flagquantum.algorithms.error_mitigation.pec import (
    run_pec as run_pec,
)

_implementation = _import_module("flagquantum.algorithms.error_mitigation.pec")
_sys.modules[__name__] = _implementation
