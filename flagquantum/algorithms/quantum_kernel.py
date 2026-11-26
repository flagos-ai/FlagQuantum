"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.quantum_ml.kernels import (
    _DEFAULT_SHOTS as _DEFAULT_SHOTS,
)
from flagquantum.algorithms.quantum_ml.kernels import (
    _MAX_FEATURE_ANGLE as _MAX_FEATURE_ANGLE,
)
from flagquantum.algorithms.quantum_ml.kernels import (
    _MAX_FEATURES as _MAX_FEATURES,
)
from flagquantum.algorithms.quantum_ml.kernels import (
    KernelMatrixResult as KernelMatrixResult,
)
from flagquantum.algorithms.quantum_ml.kernels import (
    KernelRidgeClassifier as KernelRidgeClassifier,
)
from flagquantum.algorithms.quantum_ml.kernels import __all__ as __all__
from flagquantum.algorithms.quantum_ml.kernels import (
    _ancilla_share as _ancilla_share,
)
from flagquantum.algorithms.quantum_ml.kernels import (
    _append_feature_state as _append_feature_state,
)
from flagquantum.algorithms.quantum_ml.kernels import (
    _generator as _generator,
)
from flagquantum.algorithms.quantum_ml.kernels import (
    _kernel_entry as _kernel_entry,
)
from flagquantum.algorithms.quantum_ml.kernels import (
    _swap_test_overlap as _swap_test_overlap,
)
from flagquantum.algorithms.quantum_ml.kernels import (
    _validated_labels as _validated_labels,
)
from flagquantum.algorithms.quantum_ml.kernels import (
    _validated_regularization as _validated_regularization,
)
from flagquantum.algorithms.quantum_ml.kernels import (
    _validated_ridge_operands as _validated_ridge_operands,
)
from flagquantum.algorithms.quantum_ml.kernels import (
    _validated_rows as _validated_rows,
)
from flagquantum.algorithms.quantum_ml.kernels import (
    kernel_ridge_classifier as kernel_ridge_classifier,
)
from flagquantum.algorithms.quantum_ml.kernels import (
    kernel_ridge_regression as kernel_ridge_regression,
)
from flagquantum.algorithms.quantum_ml.kernels import (
    quantum_kernel_matrix as quantum_kernel_matrix,
)

_implementation = _import_module("flagquantum.algorithms.quantum_ml.kernels")
_sys.modules[__name__] = _implementation
