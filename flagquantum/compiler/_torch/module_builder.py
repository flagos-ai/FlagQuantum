"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends.torch._module_builder import (
    _BINARY as _BINARY,
)
from flagquantum.compiler.frontends.torch._module_builder import (
    _OTHER as _OTHER,
)
from flagquantum.compiler.frontends.torch._module_builder import (
    _OVERLOADS as _OVERLOADS,
)
from flagquantum.compiler.frontends.torch._module_builder import (
    _SHAPES as _SHAPES,
)
from flagquantum.compiler.frontends.torch._module_builder import (
    _SUPPORTED as _SUPPORTED,
)
from flagquantum.compiler.frontends.torch._module_builder import (
    _UNARY as _UNARY,
)
from flagquantum.compiler.frontends.torch._module_builder import (
    CapturedModuleBuilder as CapturedModuleBuilder,
)
from flagquantum.compiler.frontends.torch._module_builder import (
    _jax_bindings as _jax_bindings,
)
from flagquantum.compiler.frontends.torch._module_builder import (
    _operation as _operation,
)
from flagquantum.compiler.frontends.torch._module_builder import (
    capture_module_builder as capture_module_builder,
)

_implementation = _import_module("flagquantum.compiler.frontends.torch._module_builder")
_sys.modules[__name__] = _implementation
