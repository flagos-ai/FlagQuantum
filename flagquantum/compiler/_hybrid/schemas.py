"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends._structured.schemas import (
    OPERATION_SCHEMAS as OPERATION_SCHEMAS,
)
from flagquantum.compiler.frontends._structured.schemas import (
    OperationSchema as OperationSchema,
)

_implementation = _import_module("flagquantum.compiler.frontends._structured.schemas")
_sys.modules[__name__] = _implementation
