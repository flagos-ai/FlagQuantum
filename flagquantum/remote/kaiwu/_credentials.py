"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.kaiwu._credentials import (
    KaiwuCredentials as KaiwuCredentials,
)
from flagquantum.compute.remote.kaiwu._credentials import (
    _normalize_pair as _normalize_pair,
)
from flagquantum.compute.remote.kaiwu._credentials import (
    resolve_kaiwu_credentials as resolve_kaiwu_credentials,
)

_implementation = _import_module("flagquantum.compute.remote.kaiwu._credentials")
_sys.modules[__name__] = _implementation
