"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.compute._resource_discovery import (
    Pages as Pages,
)
from flagquantum.compute.remote.compute._resource_discovery import (
    _high_priority_configurations as _high_priority_configurations,
)
from flagquantum.compute.remote.compute._resource_discovery import (
    _named_record as _named_record,
)
from flagquantum.compute.remote.compute._resource_discovery import (
    _objects as _objects,
)
from flagquantum.compute.remote.compute._resource_discovery import (
    _private_images as _private_images,
)
from flagquantum.compute.remote.compute._resource_discovery import (
    _Request as _Request,
)
from flagquantum.compute.remote.compute._resource_discovery import (
    discover_resources as discover_resources,
)

_implementation = _import_module(
    "flagquantum.compute.remote.compute._resource_discovery"
)
_sys.modules[__name__] = _implementation
