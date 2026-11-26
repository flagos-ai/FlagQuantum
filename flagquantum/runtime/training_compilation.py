"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.runtime.training.compilation import (
    _build_training_circuit as _build_training_circuit,
)
from flagquantum.runtime.training.compilation import (
    _compilation_diagnostics as _compilation_diagnostics,
)
from flagquantum.runtime.training.compilation import (
    _configuration as _configuration,
)
from flagquantum.runtime.training.compilation import (
    _execute_prebuilt as _execute_prebuilt,
)
from flagquantum.runtime.training.compilation import (
    _GradientCall as _GradientCall,
)
from flagquantum.runtime.training.compilation import (
    _GradientResult as _GradientResult,
)
from flagquantum.runtime.training.compilation import (
    _objective_capture_configuration as _objective_capture_configuration,
)
from flagquantum.runtime.training.compilation import (
    _objective_signature as _objective_signature,
)
from flagquantum.runtime.training.compilation import (
    _optimizer_exclusion as _optimizer_exclusion,
)
from flagquantum.runtime.training.compilation import (
    _prepare_training_program as _prepare_training_program,
)
from flagquantum.runtime.training.compilation import (
    _prepared_execution as _prepared_execution,
)
from flagquantum.runtime.training.compilation import (
    _scalar_objective as _scalar_objective,
)
from flagquantum.runtime.training.compilation import (
    _snapshot_statistics as _snapshot_statistics,
)
from flagquantum.runtime.training.compilation import (
    _tensor_signature as _tensor_signature,
)
from flagquantum.runtime.training.compilation import (
    _training_exclusion as _training_exclusion,
)
from flagquantum.runtime.training.compilation import (
    _training_signature as _training_signature,
)
from flagquantum.runtime.training.compilation import (
    _training_value as _training_value,
)
from flagquantum.runtime.training.compilation import (
    _TrainingProgram as _TrainingProgram,
)

_implementation = _import_module("flagquantum.runtime.training.compilation")
_sys.modules[__name__] = _implementation
