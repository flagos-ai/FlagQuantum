"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.realtime.protocol import (
    _FNV_MASK as _FNV_MASK,
)
from flagquantum.compute.realtime.protocol import (
    _FNV_OFFSET_BASIS as _FNV_OFFSET_BASIS,
)
from flagquantum.compute.realtime.protocol import (
    _FNV_PRIME as _FNV_PRIME,
)
from flagquantum.compute.realtime.protocol import (
    _REQUEST as _REQUEST,
)
from flagquantum.compute.realtime.protocol import (
    _RESPONSE as _RESPONSE,
)
from flagquantum.compute.realtime.protocol import (
    HEADER_SIZE as HEADER_SIZE,
)
from flagquantum.compute.realtime.protocol import (
    MAGIC_REQUEST as MAGIC_REQUEST,
)
from flagquantum.compute.realtime.protocol import (
    MAGIC_RESPONSE as MAGIC_RESPONSE,
)
from flagquantum.compute.realtime.protocol import (
    PROTOCOL_SCHEMA as PROTOCOL_SCHEMA,
)
from flagquantum.compute.realtime.protocol import (
    STATUS_SUCCESS as STATUS_SUCCESS,
)
from flagquantum.compute.realtime.protocol import (
    RealtimeCallError as RealtimeCallError,
)
from flagquantum.compute.realtime.protocol import (
    RealtimeProtocolError as RealtimeProtocolError,
)
from flagquantum.compute.realtime.protocol import (
    RPCHeader as RPCHeader,
)
from flagquantum.compute.realtime.protocol import (
    RPCResponse as RPCResponse,
)
from flagquantum.compute.realtime.protocol import __all__ as __all__
from flagquantum.compute.realtime.protocol import (
    _require_room as _require_room,
)
from flagquantum.compute.realtime.protocol import (
    _require_slot_size as _require_slot_size,
)
from flagquantum.compute.realtime.protocol import (
    frame_request as frame_request,
)
from flagquantum.compute.realtime.protocol import (
    frame_response as frame_response,
)
from flagquantum.compute.realtime.protocol import (
    function_id as function_id,
)
from flagquantum.compute.realtime.protocol import (
    read_request as read_request,
)
from flagquantum.compute.realtime.protocol import (
    read_response as read_response,
)

_implementation = _import_module("flagquantum.compute.realtime.protocol")
_sys.modules[__name__] = _implementation
