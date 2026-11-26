"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends._structured.runtime_inputs import (
    _input_descriptor as _input_descriptor,
)
from flagquantum.compiler.frontends._structured.runtime_inputs import (
    _input_signature as _input_signature,
)
from flagquantum.compiler.frontends._structured.runtime_inputs import (
    _normalized_dtype as _normalized_dtype,
)
from flagquantum.compiler.frontends._structured.runtime_inputs import (
    _runtime_device_class as _runtime_device_class,
)
from flagquantum.compiler.frontends._structured.runtime_inputs import (
    _runtime_shape as _runtime_shape,
)

_implementation = _import_module(
    "flagquantum.compiler.frontends._structured.runtime_inputs"
)
_sys.modules[__name__] = _implementation
