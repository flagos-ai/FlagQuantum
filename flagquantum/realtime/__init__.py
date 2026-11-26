"""Historical import facade for the relocated implementation."""

import sys as _alias_sys
from importlib import import_module as _import_module
from typing import Any as _Any

from flagquantum.compute.realtime import (
    DEFAULT_SLOT_SIZE as DEFAULT_SLOT_SIZE,
)
from flagquantum.compute.realtime import (
    DISPATCH_MODES as DISPATCH_MODES,
)
from flagquantum.compute.realtime import (
    HEADER_SIZE as HEADER_SIZE,
)
from flagquantum.compute.realtime import (
    MAGIC_REQUEST as MAGIC_REQUEST,
)
from flagquantum.compute.realtime import (
    MAGIC_RESPONSE as MAGIC_RESPONSE,
)
from flagquantum.compute.realtime import (
    MAX_ARGUMENTS as MAX_ARGUMENTS,
)
from flagquantum.compute.realtime import (
    MAX_RESULTS as MAX_RESULTS,
)
from flagquantum.compute.realtime import (
    PROTOCOL_SCHEMA as PROTOCOL_SCHEMA,
)
from flagquantum.compute.realtime import (
    SCHEMA_SIZE as SCHEMA_SIZE,
)
from flagquantum.compute.realtime import (
    STATUS_HANDLER_RAISED as STATUS_HANDLER_RAISED,
)
from flagquantum.compute.realtime import (
    STATUS_INVALID_ARGUMENTS as STATUS_INVALID_ARGUMENTS,
)
from flagquantum.compute.realtime import (
    STATUS_SUCCESS as STATUS_SUCCESS,
)
from flagquantum.compute.realtime import (
    STATUS_UNKNOWN_FUNCTION as STATUS_UNKNOWN_FUNCTION,
)
from flagquantum.compute.realtime import (
    TYPE_DESCRIPTOR_SIZE as TYPE_DESCRIPTOR_SIZE,
)
from flagquantum.compute.realtime import (
    ChannelFactory as ChannelFactory,
)
from flagquantum.compute.realtime import (
    DeviceCallChannel as DeviceCallChannel,
)
from flagquantum.compute.realtime import (
    DeviceCallChannelDriver as DeviceCallChannelDriver,
)
from flagquantum.compute.realtime import (
    DeviceCallOutcome as DeviceCallOutcome,
)
from flagquantum.compute.realtime import (
    DeviceCallService as DeviceCallService,
)
from flagquantum.compute.realtime import (
    DeviceCallSession as DeviceCallSession,
)
from flagquantum.compute.realtime import (
    DispatchMode as DispatchMode,
)
from flagquantum.compute.realtime import (
    FunctionEntry as FunctionEntry,
)
from flagquantum.compute.realtime import (
    FunctionTableService as FunctionTableService,
)
from flagquantum.compute.realtime import (
    FunctionTableSession as FunctionTableSession,
)
from flagquantum.compute.realtime import (
    HandlerSchema as HandlerSchema,
)
from flagquantum.compute.realtime import (
    LoopbackTransport as LoopbackTransport,
)
from flagquantum.compute.realtime import (
    PayloadType as PayloadType,
)
from flagquantum.compute.realtime import (
    RealtimeCallError as RealtimeCallError,
)
from flagquantum.compute.realtime import (
    RealtimeHandlerError as RealtimeHandlerError,
)
from flagquantum.compute.realtime import (
    RealtimeProtocolError as RealtimeProtocolError,
)
from flagquantum.compute.realtime import (
    RealtimeSession as RealtimeSession,
)
from flagquantum.compute.realtime import (
    RingTransport as RingTransport,
)
from flagquantum.compute.realtime import (
    RPCHeader as RPCHeader,
)
from flagquantum.compute.realtime import (
    RPCResponse as RPCResponse,
)
from flagquantum.compute.realtime import (
    TypeDescriptor as TypeDescriptor,
)
from flagquantum.compute.realtime import __all__ as __all__
from flagquantum.compute.realtime import (
    decode_arguments as decode_arguments,
)
from flagquantum.compute.realtime import (
    decode_results as decode_results,
)
from flagquantum.compute.realtime import (
    encode_arguments as encode_arguments,
)
from flagquantum.compute.realtime import (
    encode_results as encode_results,
)
from flagquantum.compute.realtime import (
    frame_request as frame_request,
)
from flagquantum.compute.realtime import (
    frame_response as frame_response,
)
from flagquantum.compute.realtime import (
    function_id as function_id,
)
from flagquantum.compute.realtime import (
    loopback_channel as loopback_channel,
)
from flagquantum.compute.realtime import (
    read_request as read_request,
)
from flagquantum.compute.realtime import (
    read_response as read_response,
)

_implementation = _import_module("flagquantum.compute.realtime")
__doc__ = _implementation.__doc__ or ""


def __getattr__(name: str) -> _Any:
    cached = _alias_sys.modules.get(f"{__name__}.{name}")
    if cached is not None:
        return cached
    return getattr(_implementation, name)
