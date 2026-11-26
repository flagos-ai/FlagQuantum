"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends._structured.frontend.diagnostics import (
    CaptureDiagnostic as CaptureDiagnostic,
)
from flagquantum.compiler.frontends._structured.frontend.diagnostics import (
    HybridCaptureError as HybridCaptureError,
)

_implementation = _import_module(
    "flagquantum.compiler.frontends._structured.frontend.diagnostics"
)
_sys.modules[__name__] = _implementation
