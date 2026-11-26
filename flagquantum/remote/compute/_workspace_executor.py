"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.compute._workspace_executor import (
    MAX_BATCH_ITEMS as MAX_BATCH_ITEMS,
)
from flagquantum.compute.remote.compute._workspace_executor import (
    MAX_MESSAGE_BYTES as MAX_MESSAGE_BYTES,
)
from flagquantum.compute.remote.compute._workspace_executor import (
    SCHEMA as SCHEMA,
)
from flagquantum.compute.remote.compute._workspace_executor import (
    SUPPORTED_MEASUREMENTS as SUPPORTED_MEASUREMENTS,
)
from flagquantum.compute.remote.compute._workspace_executor import (
    VERSION as VERSION,
)
from flagquantum.compute.remote.compute._workspace_executor import (
    _encode_measurement_value as _encode_measurement_value,
)
from flagquantum.compute.remote.compute._workspace_executor import (
    _encode_tensor as _encode_tensor,
)
from flagquantum.compute.remote.compute._workspace_executor import (
    _error_response as _error_response,
)
from flagquantum.compute.remote.compute._workspace_executor import (
    _read_exact as _read_exact,
)
from flagquantum.compute.remote.compute._workspace_executor import (
    execute as execute,
)
from flagquantum.compute.remote.compute._workspace_executor import (
    main as main,
)
from flagquantum.compute.remote.compute._workspace_executor import (
    read_message as read_message,
)
from flagquantum.compute.remote.compute._workspace_executor import (
    serve as serve,
)
from flagquantum.compute.remote.compute._workspace_executor import (
    write_message as write_message,
)

_implementation = _import_module(
    "flagquantum.compute.remote.compute._workspace_executor"
)
_sys.modules[__name__] = _implementation

if __name__ == "__main__":
    main()
