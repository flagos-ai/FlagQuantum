"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends._structured.rewrites import __all__ as __all__
from flagquantum.compiler.frontends._structured.rewrites import (
    clone_inline_block as clone_inline_block,
)
from flagquantum.compiler.frontends._structured.rewrites import (
    fresh_value as fresh_value,
)
from flagquantum.compiler.frontends._structured.rewrites import (
    region_operation_count as region_operation_count,
)
from flagquantum.compiler.frontends._structured.rewrites import (
    rewrite_regions as rewrite_regions,
)
from flagquantum.compiler.frontends._structured.rewrites import (
    substitute_operation as substitute_operation,
)
from flagquantum.compiler.frontends._structured.rewrites import (
    substitute_value as substitute_value,
)

_implementation = _import_module("flagquantum.compiler.frontends._structured.rewrites")
_sys.modules[__name__] = _implementation
