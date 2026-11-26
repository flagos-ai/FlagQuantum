"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.compute._jiuding_images import (
    resolve_catalog_image as resolve_catalog_image,
)

_implementation = _import_module("flagquantum.compute.remote.compute._jiuding_images")
_sys.modules[__name__] = _implementation
