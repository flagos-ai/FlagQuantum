"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends._structured.frontend.control import (
    bind_loop_target as bind_loop_target,
)
from flagquantum.compiler.frontends._structured.frontend.control import (
    capture_branch_region as capture_branch_region,
)
from flagquantum.compiler.frontends._structured.frontend.control import (
    capture_for as capture_for,
)
from flagquantum.compiler.frontends._structured.frontend.control import (
    capture_if as capture_if,
)
from flagquantum.compiler.frontends._structured.frontend.control import (
    loop_bounds as loop_bounds,
)
from flagquantum.compiler.frontends._structured.frontend.control import (
    loop_carried_names as loop_carried_names,
)
from flagquantum.compiler.frontends._structured.frontend.control import (
    reject_outer_binding_writes as reject_outer_binding_writes,
)
from flagquantum.compiler.frontends._structured.frontend.control import (
    require_classical_carry as require_classical_carry,
)
from flagquantum.compiler.frontends._structured.frontend.control import (
    require_matching_carry as require_matching_carry,
)
from flagquantum.compiler.frontends._structured.frontend.control import (
    structured_carried_names as structured_carried_names,
)

_implementation = _import_module(
    "flagquantum.compiler.frontends._structured.frontend.control"
)
_sys.modules[__name__] = _implementation
