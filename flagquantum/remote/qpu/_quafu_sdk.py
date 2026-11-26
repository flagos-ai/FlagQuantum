"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.qpu._quafu_sdk import (
    _DISTRIBUTION as _DISTRIBUTION,
)
from flagquantum.compute.remote.qpu._quafu_sdk import (
    _EXTRA as _EXTRA,
)
from flagquantum.compute.remote.qpu._quafu_sdk import __all__ as __all__
from flagquantum.compute.remote.qpu._quafu_sdk import (
    _load_quafu_sdk as _load_quafu_sdk,
)
from flagquantum.compute.remote.qpu._quafu_sdk import (
    _plain_mapping as _plain_mapping,
)
from flagquantum.compute.remote.qpu._quafu_sdk import (
    _plain_value as _plain_value,
)
from flagquantum.compute.remote.qpu._quafu_sdk import (
    _QuafuSdkAdapter as _QuafuSdkAdapter,
)
from flagquantum.compute.remote.qpu._quafu_sdk import (
    _raise_sdk_failure as _raise_sdk_failure,
)
from flagquantum.compute.remote.qpu._quafu_sdk import (
    _submission_api_key as _submission_api_key,
)

_implementation = _import_module("flagquantum.compute.remote.qpu._quafu_sdk")
_sys.modules[__name__] = _implementation
