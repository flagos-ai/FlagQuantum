"""Retained entry points for private hybrid Python capture."""

from flagquantum._import_compat import alias_module as _alias_module
from flagquantum.compiler.frontends._structured.frontend.diagnostics import (
    CaptureDiagnostic as CaptureDiagnostic,
)
from flagquantum.compiler.frontends._structured.frontend.diagnostics import (
    HybridCaptureError as HybridCaptureError,
)
from flagquantum.compiler.frontends._structured.frontend.source import (
    capture_function as capture_function,
)
from flagquantum.compiler.frontends._structured.frontend.source import (
    capture_source as capture_source,
)

_alias_module("flagquantum.compiler._hybrid.capture", __name__)
