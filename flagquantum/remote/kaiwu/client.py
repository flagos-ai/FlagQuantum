"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.kaiwu.client import (
    _MAX_SCHEMA_DIMENSIONS as _MAX_SCHEMA_DIMENSIONS,
)
from flagquantum.compute.remote.kaiwu.client import (
    _MAX_SCHEMA_FIELDS as _MAX_SCHEMA_FIELDS,
)
from flagquantum.compute.remote.kaiwu.client import (
    _MAX_SCHEMA_SEQUENCE_TYPES as _MAX_SCHEMA_SEQUENCE_TYPES,
)
from flagquantum.compute.remote.kaiwu.client import (
    _PINNED_1_3_1_TASK_MODES as _PINNED_1_3_1_TASK_MODES,
)
from flagquantum.compute.remote.kaiwu.client import (
    _PINNED_SDK_VERSION as _PINNED_SDK_VERSION,
)
from flagquantum.compute.remote.kaiwu.client import (
    _SAFE_SCHEMA_NAME as _SAFE_SCHEMA_NAME,
)
from flagquantum.compute.remote.kaiwu.client import (
    KaiwuSDKClient as KaiwuSDKClient,
)

_implementation = _import_module("flagquantum.compute.remote.kaiwu.client")
_sys.modules[__name__] = _implementation
