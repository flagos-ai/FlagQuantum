"""Historical deployment import facade."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.runtime.dynamic.remote_sealing import _required_text as _required_text
from flagquantum.runtime.dynamic.remote_sealing import (
    _seal_dynamic_remote_package as _seal_dynamic_remote_package,
)
from flagquantum.runtime.dynamic.remote_sealing import _seal_payload as _seal_payload
from flagquantum.runtime.dynamic.remote_sealing import (
    _validate_dynamic_remote_package_seal as _validate_dynamic_remote_package_seal,
)

_sys.modules[__name__] = _import_module("flagquantum.runtime.dynamic.remote_sealing")
