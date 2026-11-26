"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.compute._distributed_job import (
    _MAX_RESULT as _MAX_RESULT,
)
from flagquantum.compute.remote.compute._distributed_job import (
    _emit_result as _emit_result,
)
from flagquantum.compute.remote.compute._distributed_job import (
    _response as _response,
)
from flagquantum.compute.remote.compute._distributed_job import (
    main as main,
)

_implementation = _import_module("flagquantum.compute.remote.compute._distributed_job")
_sys.modules[__name__] = _implementation

if __name__ == "__main__":
    main()
