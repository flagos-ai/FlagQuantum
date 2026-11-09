"""Device-call channels: how a request reaches a service and a response returns.

A realtime device call is resolved at runtime, not compiled in: a program names
a device id and a handler, and the channel that carries the request is chosen
when the application starts. This module owns that choice and the channel
lifecycle around it. Every channel moves one framed slot out and one framed slot
back, so a channel that has both halves of a ring buffer is a channel this
module can drive.

**What is not here.** No network transport and no clock. `LoopbackTransport` is a
software stand-in that performs no I/O and is how the protocol is exercised
offline; a real channel is an FPGA or NIC implementation and is not part of this
package. There is deliberately no per-dispatch timeout: this module has no clock
to measure one with, so waiting belongs to the transport, and adding a field
nothing could test would be a claim rather than a behaviour.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .protocol import (
    RealtimeCallError,
    RealtimeProtocolError,
    RPCHeader,
    RPCResponse,
    frame_request,
    frame_response,
    read_request,
    read_response,
)

if TYPE_CHECKING:
    from .service import DeviceCallSession

__all__ = [
    "DEFAULT_SLOT_SIZE",
    "ChannelFactory",
    "DeviceCallChannel",
    "DeviceCallChannelDriver",
    "DeviceCallOutcome",
    "LoopbackTransport",
    "RingTransport",
    "loopback_channel",
]

#: One slot holds a 24-byte header and the payload behind it. The reference
#: configuration uses 384-byte slots; a caller may choose any size that leaves
#: room, because the slot size is geometry rather than protocol.
DEFAULT_SLOT_SIZE = 384


class RingTransport(Protocol):
    """The transport half of a channel: an RX path and a TX path."""

    def publish_request(self, slot: bytes) -> None:
        """Hand one framed request slot to the service side of the ring."""
        ...

    def await_response(self) -> bytes:
        """Return the next framed response slot, blocking if the design waits."""
        ...

    def close(self) -> None:
        """Release whatever the transport holds."""
        ...


@dataclass(frozen=True, slots=True)
class DeviceCallOutcome:
    """One response header and the result payload that followed it."""

    response: RPCResponse
    result: bytes

    @property
    def succeeded(self) -> bool:
        """Whether the dispatcher reported success rather than a failure."""
        return self.response.succeeded


class LoopbackTransport:
    """A ring that dispatches in process instead of over a network.

    This is the offline stand-in for a NIC: it frames nothing itself, performs
    no I/O, and exists so the protocol can be driven end to end without
    hardware. It is not a transport a deployment should use, and it makes no
    latency claim.
    """

    def __init__(self, session: DeviceCallSession, *, slot_size: int) -> None:
        self._session = session
        self._slot_size = slot_size
        self._responses: deque[bytes] = deque()
        self._processed = 0
        self._closed = False

    @property
    def processed(self) -> int:
        """How many request slots this ring has dispatched."""
        return self._processed

    def publish_request(self, slot: bytes) -> None:
        """Dispatch `slot` now and queue the framed response it produced.

        Framing stays here rather than in the service: ring geometry is the
        channel's business, so the service answers with a header and a result
        payload and this transport lays them into a slot.
        """
        if self._closed:
            raise RealtimeCallError("the loopback ring is closed")
        header, payload = read_request(slot)
        response, result = self._session.dispatch(header, payload)
        self._responses.append(
            frame_response(response, result, slot_size=self._slot_size)
        )
        self._processed += 1

    def await_response(self) -> bytes:
        """Return the next queued response slot."""
        if not self._responses:
            raise RealtimeCallError(
                "the loopback ring holds no response: a response is only queued "
                "by publishing a request first"
            )
        return self._responses.popleft()

    def close(self) -> None:
        """Mark the ring closed and drop anything still queued."""
        self._closed = True
        self._responses.clear()


class DeviceCallChannel:
    """One named way to carry a device call, opened per session.

    A channel does not know which handler it is reaching or what the payload
    means; it moves slots and checks that the response belongs to the request it
    answered.
    """

    def __init__(self, name: str, transport: RingTransport, *, slot_size: int) -> None:
        if not isinstance(name, str) or not name.strip():
            raise RealtimeCallError(
                "a device-call channel name must be a non-empty string"
            )
        self._name = name
        self._transport = transport
        self._slot_size = slot_size
        self._open = False

    @property
    def name(self) -> str:
        """The name a command line would select this channel by."""
        return self._name

    @property
    def slot_size(self) -> int:
        """The fixed size of one request or response slot on this channel."""
        return self._slot_size

    @property
    def is_open(self) -> bool:
        """Whether the channel is ready to carry a call."""
        return self._open

    def open(self) -> None:
        """Start carrying calls, refusing to open an already-open channel."""
        if self._open:
            raise RealtimeCallError(f"channel {self._name!r} is already open")
        self._open = True

    def close(self) -> None:
        """Stop carrying calls and release the transport.

        Closing an already-closed channel is a no-op so teardown can be
        unconditional, unlike opening, because a second open would silently
        discard the first channel's state.
        """
        if not self._open:
            return
        self._open = False
        self._transport.close()

    def call(
        self,
        function_id: int,
        arguments: bytes = b"",
        *,
        request_id: int = 0,
        ptp_timestamp: int = 0,
    ) -> DeviceCallOutcome:
        """Send one call and return the response that answered it.

        `request_id` and `ptp_timestamp` are echoed by the service, so a
        mismatch means the response answers a different request and is refused
        rather than returned.
        """
        if not self._open:
            raise RealtimeCallError(f"channel {self._name!r} is not open")
        header = RPCHeader(function_id, len(arguments), request_id, ptp_timestamp)
        self._transport.publish_request(
            frame_request(header, arguments, slot_size=self._slot_size)
        )
        response, result = read_response(self._transport.await_response())
        if response.request_id != request_id:
            raise RealtimeProtocolError(
                f"channel {self._name!r} sent request id {request_id} but the "
                f"response echoes {response.request_id}"
            )
        if response.ptp_timestamp != ptp_timestamp:
            raise RealtimeProtocolError(
                f"channel {self._name!r} sent timestamp {ptp_timestamp} but the "
                f"response echoes {response.ptp_timestamp}"
            )
        return DeviceCallOutcome(response, result)


#: A channel factory closes over one channel's geometry and transport, so the
#: driver never has to know which channel it is building.
ChannelFactory = Callable[[], DeviceCallChannel]


class DeviceCallChannelDriver:
    """The set of channels a program may select at run time.

    Registration is explicit and a name is registered once. Opening a name
    nobody registered names the ones that exist, because a mis-typed channel on
    a command line should be a one-line correction rather than a hunt.
    """

    def __init__(self, factories: Mapping[str, ChannelFactory] | None = None) -> None:
        self._factories: dict[str, ChannelFactory] = {}
        for name, factory in (factories or {}).items():
            self.register(name, factory)

    @property
    def names(self) -> tuple[str, ...]:
        """The registered channel names, in registration order."""
        return tuple(self._factories)

    def register(self, name: str, factory: ChannelFactory) -> None:
        """Add a channel under `name`, refusing a name already in use."""
        if not isinstance(name, str) or not name.strip():
            raise RealtimeCallError(
                "a device-call channel name must be a non-empty string"
            )
        if name in self._factories:
            raise RealtimeCallError(f"channel {name!r} is already registered")
        if not callable(factory):
            raise RealtimeCallError(f"channel {name!r} needs a callable factory")
        self._factories[name] = factory

    def open(self, name: str) -> DeviceCallChannel:
        """Build and open the channel registered as `name`."""
        if name not in self._factories:
            known = ", ".join(sorted(self._factories)) or "none"
            raise RealtimeCallError(
                f"no device-call channel is registered as {name!r}; registered "
                f"channels are {known}"
            )
        channel = self._factories[name]()
        channel.open()
        return channel


def loopback_channel(
    name: str,
    session: DeviceCallSession,
    *,
    slot_size: int = DEFAULT_SLOT_SIZE,
) -> Callable[[], DeviceCallChannel]:
    """Return a factory for an in-process channel over `session`.

    No slot count is taken: a ring's slot count bounds how many requests can be
    in flight at once, and an in-process ring dispatches each request as it
    arrives, so a count here would be a number with no effect.
    """

    def factory() -> DeviceCallChannel:
        return DeviceCallChannel(
            name, LoopbackTransport(session, slot_size=slot_size), slot_size=slot_size
        )

    return factory
