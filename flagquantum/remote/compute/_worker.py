"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.compute._worker import (
    main as main,
)

_implementation = _import_module("flagquantum.compute.remote.compute._worker")
_sys.modules[__name__] = _implementation

if __name__ == "__main__":
    main()
