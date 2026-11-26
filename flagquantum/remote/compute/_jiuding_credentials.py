"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.compute._jiuding_credentials import (
    JiudingCredentials as JiudingCredentials,
)
from flagquantum.compute.remote.compute._jiuding_credentials import (
    _decode as _decode,
)
from flagquantum.compute.remote.compute._jiuding_credentials import (
    load_jiuding_credentials as load_jiuding_credentials,
)
from flagquantum.compute.remote.compute._jiuding_credentials import (
    resolve_jiuding_credentials as resolve_jiuding_credentials,
)

_implementation = _import_module(
    "flagquantum.compute.remote.compute._jiuding_credentials"
)
_sys.modules[__name__] = _implementation
