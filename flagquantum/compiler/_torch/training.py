"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends.torch._training import (
    _P as _P,
)
from flagquantum.compiler.frontends.torch._training import (
    _R as _R,
)
from flagquantum.compiler.frontends.torch._training import (
    TorchBackend as TorchBackend,
)
from flagquantum.compiler.frontends.torch._training import (
    _functional_tensor_operation as _functional_tensor_operation,
)
from flagquantum.compiler.frontends.torch._training import (
    _objective_exclusion as _objective_exclusion,
)
from flagquantum.compiler.frontends.torch._training import (
    _safe_objective_capture as _safe_objective_capture,
)
from flagquantum.compiler.frontends.torch._training import (
    compile_training_callable as compile_training_callable,
)

_implementation = _import_module("flagquantum.compiler.frontends.torch._training")
_sys.modules[__name__] = _implementation
