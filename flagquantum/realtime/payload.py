"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.realtime.payload import (
    _ELEMENT_FORMAT as _ELEMENT_FORMAT,
)
from flagquantum.compute.realtime.payload import (
    _ELEMENT_WIDTH as _ELEMENT_WIDTH,
)
from flagquantum.compute.realtime.payload import (
    _SCALAR_STRUCT as _SCALAR_STRUCT,
)
from flagquantum.compute.realtime.payload import (
    _SCALAR_WIDTH as _SCALAR_WIDTH,
)
from flagquantum.compute.realtime.payload import (
    _SCHEMA_HEADER as _SCHEMA_HEADER,
)
from flagquantum.compute.realtime.payload import (
    _SCHEMA_HEADER_SIZE as _SCHEMA_HEADER_SIZE,
)
from flagquantum.compute.realtime.payload import (
    _TYPE_DESCRIPTOR as _TYPE_DESCRIPTOR,
)
from flagquantum.compute.realtime.payload import (
    MAX_ARGUMENTS as MAX_ARGUMENTS,
)
from flagquantum.compute.realtime.payload import (
    MAX_RESULTS as MAX_RESULTS,
)
from flagquantum.compute.realtime.payload import (
    SCHEMA_SIZE as SCHEMA_SIZE,
)
from flagquantum.compute.realtime.payload import (
    TYPE_DESCRIPTOR_SIZE as TYPE_DESCRIPTOR_SIZE,
)
from flagquantum.compute.realtime.payload import (
    HandlerSchema as HandlerSchema,
)
from flagquantum.compute.realtime.payload import (
    PayloadType as PayloadType,
)
from flagquantum.compute.realtime.payload import (
    TypeDescriptor as TypeDescriptor,
)
from flagquantum.compute.realtime.payload import __all__ as __all__
from flagquantum.compute.realtime.payload import (
    _decode as _decode,
)
from flagquantum.compute.realtime.payload import (
    _decode_one as _decode_one,
)
from flagquantum.compute.realtime.payload import (
    _encode as _encode,
)
from flagquantum.compute.realtime.payload import (
    _encode_array as _encode_array,
)
from flagquantum.compute.realtime.payload import (
    _encode_one as _encode_one,
)
from flagquantum.compute.realtime.payload import (
    _encode_scalar as _encode_scalar,
)
from flagquantum.compute.realtime.payload import (
    decode_arguments as decode_arguments,
)
from flagquantum.compute.realtime.payload import (
    decode_results as decode_results,
)
from flagquantum.compute.realtime.payload import (
    encode_arguments as encode_arguments,
)
from flagquantum.compute.realtime.payload import (
    encode_results as encode_results,
)

_implementation = _import_module("flagquantum.compute.realtime.payload")
_sys.modules[__name__] = _implementation
