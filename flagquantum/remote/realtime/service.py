"""The service side of a device call: a function table and schema-driven parsing.

A device call names a handler by a hashed function id, and the bytes that follow
the header are only meaningful once a registered schema says what they are. This
module owns that side: the table, the schema-driven decode of a request and
encode of a response, and the status semantics of a call that cannot be served.

The protocol has no runtime type check on the wire, so the schema is the whole
contract. A handler whose declared schema disagrees with its caller's is a
defect in the *declaration*, and this module's job is to make the disagreement
visible in a status rather than to reinterpret the bytes.

**What is not here.** No channel, no ring, no transport, and no timing. A session
reports a status and, on failure, echoes the request id so a caller can still
match the answer. Signalling a handler-specific failure is
:class:`RealtimeHandlerError`; an unclassified exception becomes a protocol-level
status rather than a fabricated success.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from .payload import (
    HandlerSchema,
    decode_arguments,
    encode_results,
)
from .protocol import (
    STATUS_SUCCESS,
    RealtimeCallError,
    RealtimeProtocolError,
    RPCHeader,
    RPCResponse,
    function_id,
)

__all__ = [
    "DISPATCH_MODES",
    "STATUS_HANDLER_RAISED",
    "STATUS_INVALID_ARGUMENTS",
    "STATUS_UNKNOWN_FUNCTION",
    "DispatchMode",
    "DeviceCallService",
    "DeviceCallSession",
    "FunctionEntry",
    "FunctionTableService",
    "FunctionTableSession",
    "RealtimeHandlerError",
]

#: The dispatch modes the realtime service interface distinguishes. A session
#: serves exactly one of them, and a service that does not support a requested
#: mode returns nothing rather than degrading to another mode.
DispatchMode = Literal["host_call", "graph_launch"]
DISPATCH_MODES: tuple[DispatchMode, ...] = ("host_call", "graph_launch")

#: Status is zero on success, positive for a handler-specific failure, and
#: negative for a protocol-level one. The negative codes below are the protocol
#: failures this dispatcher can report: it never invents success.
STATUS_UNKNOWN_FUNCTION = -1
STATUS_INVALID_ARGUMENTS = -2
STATUS_HANDLER_RAISED = -3


class RealtimeHandlerError(RuntimeError):
    """A handler refusing a request it understood.

    The status must be positive, because that is the protocol's range for a
    handler-specific failure; a handler cannot claim a protocol-level code.
    """

    def __init__(self, status: int, message: str) -> None:
        if type(status) is not int or not 0 < status < 2**31:
            raise RealtimeCallError(
                f"a handler failure status must be a positive signed 32-bit "
                f"integer, not {status!r}"
            )
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, slots=True)
class FunctionEntry:
    """One handler: what it is called, what it accepts, and what it does.

    `invoke` receives the decoded arguments, one per argument descriptor, and
    returns the decoded results. Decoding and encoding stay here so a handler
    never touches the wire format.
    """

    name: str
    schema: HandlerSchema
    invoke: Callable[[tuple[object, ...]], object]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise RealtimeCallError("a function entry needs a non-empty handler name")
        if not isinstance(self.schema, HandlerSchema):
            raise RealtimeCallError("a function entry needs a HandlerSchema")
        if not callable(self.invoke):
            raise RealtimeCallError("a function entry needs a callable handler")

    @property
    def function_id(self) -> int:
        """The id a compiled call site names this handler by."""
        return function_id(self.name)


class DeviceCallSession(Protocol):
    """One owned service session, serving one dispatch mode."""

    @property
    def mode(self) -> DispatchMode:
        """The dispatch mode this session was opened for."""
        ...

    def function_table(self) -> Mapping[int, FunctionEntry]:
        """The live table, keyed by function id."""
        ...

    def dispatch(self, header: RPCHeader, payload: bytes) -> tuple[RPCResponse, bytes]:
        """Serve one request and return the response header and result bytes."""
        ...

    def stop(self) -> None:
        """Release the table and anything built around it."""
        ...


class DeviceCallService(Protocol):
    """A provider of device-call sessions, one per requested dispatch mode."""

    def open_session(self, mode: DispatchMode) -> DeviceCallSession | None:
        """Return a session for `mode`, or none if this service cannot serve it."""
        ...


class FunctionTableSession:
    """A session over a fixed set of handlers, valid until it is stopped."""

    def __init__(self, mode: DispatchMode, table: Mapping[int, FunctionEntry]) -> None:
        self._mode = mode
        self._table: dict[int, FunctionEntry] = dict(table)
        self._stopped = False

    @property
    def mode(self) -> DispatchMode:
        """The dispatch mode this session was opened for."""
        return self._mode

    def function_table(self) -> Mapping[int, FunctionEntry]:
        """Return the live table.

        The table is owned by the session and stays readable for as long as the
        session is live, so a caller may hold onto it; after `stop()` it is gone
        and asking for it again is a lifecycle mistake rather than an empty
        answer.
        """
        if self._stopped:
            raise RealtimeCallError(
                "the device-call session is stopped and no longer owns a function table"
            )
        return self._table

    def dispatch(self, header: RPCHeader, payload: bytes) -> tuple[RPCResponse, bytes]:
        """Serve one request, echoing the request id and timestamp on every path."""
        if self._stopped:
            raise RealtimeCallError("the device-call session is stopped")
        entry = self._table.get(header.function_id)
        if entry is None:
            return self._fail(STATUS_UNKNOWN_FUNCTION, header)
        try:
            arguments = decode_arguments(entry.schema, payload)
        except RealtimeProtocolError:
            return self._fail(STATUS_INVALID_ARGUMENTS, header)
        try:
            results = entry.invoke(arguments)
        except RealtimeHandlerError as failure:
            return self._fail(failure.status, header)
        except Exception:
            # A dispatcher sits on a ring buffer: an escaping exception would
            # leave a slot without a response. The status says "unclassified",
            # so the failure is visible rather than reported as success.
            return self._fail(STATUS_HANDLER_RAISED, header)
        return self._succeed(entry.schema, results, header)

    def stop(self) -> None:
        """Release the table. Stopping twice is a no-op."""
        self._stopped = True
        self._table = {}

    def _succeed(
        self, schema: HandlerSchema, results: object, header: RPCHeader
    ) -> tuple[RPCResponse, bytes]:
        values = _as_results(schema, results)
        try:
            payload = encode_results(schema, values)
        except RealtimeProtocolError:
            return self._fail(STATUS_HANDLER_RAISED, header)
        return (
            RPCResponse(
                STATUS_SUCCESS, len(payload), header.request_id, header.ptp_timestamp
            ),
            payload,
        )

    @staticmethod
    def _fail(status: int, header: RPCHeader) -> tuple[RPCResponse, bytes]:
        return RPCResponse(status, 0, header.request_id, header.ptp_timestamp), b""


def _as_results(schema: HandlerSchema, results: object) -> Sequence[object]:
    """Read a handler's return value as one result per result descriptor."""
    if isinstance(results, Sequence) and not isinstance(results, str):
        return results
    if schema.num_results == 1:
        return (results,)
    raise RealtimeProtocolError(
        f"the handler declares {schema.num_results} results and must return a "
        f"sequence with one value per result"
    )


class FunctionTableService:
    """A service built from a fixed set of handlers and the modes it serves.

    Building the table hashes every handler name, so two names that collide
    under the function-id hash are refused here. A collision would make a call
    reach the wrong handler, and the protocol has no field that could tell the
    caller so.
    """

    def __init__(
        self,
        entries: Sequence[FunctionEntry],
        *,
        modes: Sequence[DispatchMode] = ("host_call",),
    ) -> None:
        if not entries:
            raise RealtimeCallError(
                "a function table service needs at least one handler"
            )
        for mode in modes:
            if mode not in DISPATCH_MODES:
                raise RealtimeCallError(
                    f"dispatch mode {mode!r} is not one of {', '.join(DISPATCH_MODES)}"
                )
        table: dict[int, FunctionEntry] = {}
        for entry in entries:
            existing = table.get(entry.function_id)
            if existing is not None:
                raise RealtimeCallError(
                    f"handlers {existing.name!r} and {entry.name!r} collide on "
                    f"function id {entry.function_id}"
                )
            table[entry.function_id] = entry
        self._table = table
        self._entries: tuple[FunctionEntry, ...] = tuple(entries)
        self._modes: tuple[DispatchMode, ...] = tuple(modes)

    @property
    def modes(self) -> tuple[DispatchMode, ...]:
        """The dispatch modes this service can open a session for."""
        return self._modes

    @property
    def entries(self) -> tuple[FunctionEntry, ...]:
        """The handlers this service declares."""
        return self._entries

    def open_session(self, mode: DispatchMode) -> DeviceCallSession | None:
        """Open a session for `mode`, or return none for a mode it cannot serve."""
        if mode not in DISPATCH_MODES:
            raise RealtimeCallError(
                f"dispatch mode {mode!r} is not one of {', '.join(DISPATCH_MODES)}"
            )
        if mode not in self._modes:
            return None
        return FunctionTableSession(mode, self._table)
