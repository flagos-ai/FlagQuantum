"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends._structured.verifier import (
    Diagnostic as Diagnostic,
)
from flagquantum.compiler.frontends._structured.verifier import (
    HybridVerificationError as HybridVerificationError,
)
from flagquantum.compiler.frontends._structured.verifier import (
    _arity_ok as _arity_ok,
)
from flagquantum.compiler.frontends._structured.verifier import (
    _is_scalar as _is_scalar,
)
from flagquantum.compiler.frontends._structured.verifier import (
    _Verifier as _Verifier,
)
from flagquantum.compiler.frontends._structured.verifier import (
    verify_program as verify_program,
)

_implementation = _import_module("flagquantum.compiler.frontends._structured.verifier")
_sys.modules[__name__] = _implementation
