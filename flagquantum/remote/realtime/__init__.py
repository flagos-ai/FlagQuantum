"""Software side of the realtime device-call protocol.

CUDA-Q's realtime surface lets a running program call a classical service on a
co-located decoder or an FPGA. Most of that surface is hardware: a ConnectX NIC,
a GPUDirect transceiver, an FPGA decode loop. The software half is a protocol
definition and a device-call abstraction, and this package owns that half so a
hardware partner has something to implement against.

Three layers, in the order a call moves through them:

- :mod:`.protocol` frames one request or response into a fixed ring slot: a
  24-byte header, an opaque payload, zero padding.
- :mod:`.payload` types that payload. The wire is typeless, so a
  :class:`~flagquantum.remote.realtime.payload.HandlerSchema` is the whole
  contract between a caller and a handler.
- :mod:`.service` is the handler side, :mod:`.channels` moves slots, and
  :mod:`.session` resolves a device id to an open channel.

**What is not here.** No NIC, no FPGA, no GPUDirect, no RDMA, no clock, and no
latency evidence. `LoopbackTransport` performs no I/O and is how the protocol is
exercised offline; a real channel is a hardware implementation that is out of
scope for this repository, and the microsecond figures in the upstream
documentation are not repeated here as a promise. This package makes no
scalability or hardware claim.
"""

from .channels import (
    DEFAULT_SLOT_SIZE,
    ChannelFactory,
    DeviceCallChannel,
    DeviceCallChannelDriver,
    DeviceCallOutcome,
    LoopbackTransport,
    RingTransport,
    loopback_channel,
)
from .payload import (
    MAX_ARGUMENTS,
    MAX_RESULTS,
    SCHEMA_SIZE,
    TYPE_DESCRIPTOR_SIZE,
    HandlerSchema,
    PayloadType,
    TypeDescriptor,
    decode_arguments,
    decode_results,
    encode_arguments,
    encode_results,
)
from .protocol import (
    HEADER_SIZE,
    MAGIC_REQUEST,
    MAGIC_RESPONSE,
    PROTOCOL_SCHEMA,
    STATUS_SUCCESS,
    RealtimeCallError,
    RealtimeProtocolError,
    RPCHeader,
    RPCResponse,
    frame_request,
    frame_response,
    function_id,
    read_request,
    read_response,
)
from .service import (
    DISPATCH_MODES,
    STATUS_HANDLER_RAISED,
    STATUS_INVALID_ARGUMENTS,
    STATUS_UNKNOWN_FUNCTION,
    DeviceCallService,
    DeviceCallSession,
    DispatchMode,
    FunctionEntry,
    FunctionTableService,
    FunctionTableSession,
    RealtimeHandlerError,
)
from .session import RealtimeSession

__all__ = [
    "DEFAULT_SLOT_SIZE",
    "DISPATCH_MODES",
    "HEADER_SIZE",
    "MAGIC_REQUEST",
    "MAGIC_RESPONSE",
    "MAX_ARGUMENTS",
    "MAX_RESULTS",
    "PROTOCOL_SCHEMA",
    "SCHEMA_SIZE",
    "STATUS_HANDLER_RAISED",
    "STATUS_INVALID_ARGUMENTS",
    "STATUS_SUCCESS",
    "STATUS_UNKNOWN_FUNCTION",
    "TYPE_DESCRIPTOR_SIZE",
    "ChannelFactory",
    "DeviceCallChannel",
    "DeviceCallChannelDriver",
    "DeviceCallOutcome",
    "DeviceCallService",
    "DeviceCallSession",
    "DispatchMode",
    "FunctionEntry",
    "FunctionTableService",
    "FunctionTableSession",
    "HandlerSchema",
    "LoopbackTransport",
    "PayloadType",
    "RPCHeader",
    "RPCResponse",
    "RealtimeCallError",
    "RealtimeHandlerError",
    "RealtimeProtocolError",
    "RealtimeSession",
    "RingTransport",
    "TypeDescriptor",
    "decode_arguments",
    "decode_results",
    "encode_arguments",
    "encode_results",
    "frame_request",
    "frame_response",
    "function_id",
    "loopback_channel",
    "read_request",
    "read_response",
]
