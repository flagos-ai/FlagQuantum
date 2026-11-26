"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.compute._workspace_discovery import __all__ as __all__
from flagquantum.compute.remote.compute._workspace_discovery import (
    select_workspace as select_workspace,
)
from flagquantum.compute.remote.compute._workspace_discovery import (
    summarize_workspaces as summarize_workspaces,
)

_implementation = _import_module(
    "flagquantum.compute.remote.compute._workspace_discovery"
)
_sys.modules[__name__] = _implementation
