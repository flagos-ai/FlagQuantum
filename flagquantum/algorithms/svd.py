"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.linear_algebra.svd import (
    _DEFAULT_SHOTS as _DEFAULT_SHOTS,
)
from flagquantum.algorithms.linear_algebra.svd import (
    _MAX_DATA_QUBITS as _MAX_DATA_QUBITS,
)
from flagquantum.algorithms.linear_algebra.svd import (
    SingularValueResult as SingularValueResult,
)
from flagquantum.algorithms.linear_algebra.svd import __all__ as __all__
from flagquantum.algorithms.linear_algebra.svd import (
    _append_block_encoding as _append_block_encoding,
)
from flagquantum.algorithms.linear_algebra.svd import (
    _counting_distribution as _counting_distribution,
)
from flagquantum.algorithms.linear_algebra.svd import (
    _embedding_of as _embedding_of,
)
from flagquantum.algorithms.linear_algebra.svd import (
    _estimation_circuit as _estimation_circuit,
)
from flagquantum.algorithms.linear_algebra.svd import (
    _input_amplitudes as _input_amplitudes,
)
from flagquantum.algorithms.linear_algebra.svd import (
    _PhaseFromEmbedding as _PhaseFromEmbedding,
)
from flagquantum.algorithms.linear_algebra.svd import (
    _select_unitary as _select_unitary,
)
from flagquantum.algorithms.linear_algebra.svd import (
    _validated_counting_width as _validated_counting_width,
)
from flagquantum.algorithms.linear_algebra.svd import (
    _validated_shots as _validated_shots,
)
from flagquantum.algorithms.linear_algebra.svd import (
    estimate_singular_values as estimate_singular_values,
)

_implementation = _import_module("flagquantum.algorithms.linear_algebra.svd")
_sys.modules[__name__] = _implementation
