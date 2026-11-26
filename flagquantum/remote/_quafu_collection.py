"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote._quafu_collection import (
    _SCHEMA as _SCHEMA,
)
from flagquantum.compute.remote._quafu_collection import (
    _CollectionReceipt as _CollectionReceipt,
)
from flagquantum.compute.remote._quafu_collection import (
    _digest as _digest,
)
from flagquantum.compute.remote._quafu_collection import (
    _fetch_results as _fetch_results,
)
from flagquantum.compute.remote._quafu_collection import (
    _fetch_sdk_results as _fetch_sdk_results,
)
from flagquantum.compute.remote._quafu_collection import (
    collection_result as collection_result,
)
from flagquantum.compute.remote._quafu_collection import (
    submit_collection as submit_collection,
)
from flagquantum.compute.remote._quafu_collection import (
    validate_context as validate_context,
)

_implementation = _import_module("flagquantum.compute.remote._quafu_collection")
_sys.modules[__name__] = _implementation
