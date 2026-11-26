"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.qpu.quafu_dynamic import (
    _DYNAMIC_FIELDS as _DYNAMIC_FIELDS,
)
from flagquantum.compute.remote.qpu.quafu_dynamic import (
    _FIXTURE_SCHEMA_VERSION as _FIXTURE_SCHEMA_VERSION,
)
from flagquantum.compute.remote.qpu.quafu_dynamic import (
    _FIXTURE_SCOPE as _FIXTURE_SCOPE,
)
from flagquantum.compute.remote.qpu.quafu_dynamic import (
    _QUAFU_DYNAMIC_EXTENSION as _QUAFU_DYNAMIC_EXTENSION,
)
from flagquantum.compute.remote.qpu.quafu_dynamic import __all__ as __all__
from flagquantum.compute.remote.qpu.quafu_dynamic import (
    _boolean as _boolean,
)
from flagquantum.compute.remote.qpu.quafu_dynamic import (
    _closed_mapping as _closed_mapping,
)
from flagquantum.compute.remote.qpu.quafu_dynamic import (
    _directed_edges as _directed_edges,
)
from flagquantum.compute.remote.qpu.quafu_dynamic import (
    _integer_tuple as _integer_tuple,
)
from flagquantum.compute.remote.qpu.quafu_dynamic import (
    _positive_int as _positive_int,
)
from flagquantum.compute.remote.qpu.quafu_dynamic import (
    _quafu_dynamic_snapshot_from_fixture as _quafu_dynamic_snapshot_from_fixture,
)
from flagquantum.compute.remote.qpu.quafu_dynamic import (
    _string_tuple as _string_tuple,
)
from flagquantum.compute.remote.qpu.quafu_dynamic import (
    _text as _text,
)
from flagquantum.compute.remote.qpu.quafu_dynamic import (
    _timestamp as _timestamp,
)

_implementation = _import_module("flagquantum.compute.remote.qpu.quafu_dynamic")
_sys.modules[__name__] = _implementation
