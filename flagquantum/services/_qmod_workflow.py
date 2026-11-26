"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.ecosystem._qmod.workflow import (
    QmodWorkflowResult as QmodWorkflowResult,
)
from flagquantum.ecosystem._qmod.workflow import (
    VqeEvaluation as VqeEvaluation,
)
from flagquantum.ecosystem._qmod.workflow import (
    VqeRun as VqeRun,
)
from flagquantum.ecosystem._qmod.workflow import (
    _EvaluationLimitError as _EvaluationLimitError,
)
from flagquantum.ecosystem._qmod.workflow import (
    run_qaoa_workflow as run_qaoa_workflow,
)

_implementation = _import_module("flagquantum.ecosystem._qmod.workflow")
_sys.modules[__name__] = _implementation
