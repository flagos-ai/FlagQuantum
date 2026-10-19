"""A realtime session: the channels a program can reach, keyed by device id.

A compiled call site names a device id and a handler, and the channel that
carries it is chosen when the program starts. This module owns that resolution:
one open channel per device id, selected from the channels the driver knows, and
refused by name when a device id nobody opened is used.

**What is not here.** No clock, no thread, no asyncio, and no timing claim. A
session resolves and carries calls; it does not decide when they happen.
"""

from __future__ import annotations

from types import TracebackType

from .channels import DeviceCallChannel, DeviceCallChannelDriver, DeviceCallOutcome
from .protocol import RealtimeCallError

__all__ = ["RealtimeSession"]


class RealtimeSession:
    """The open device-call channels of one application run.

    A session is a context manager, because closing it is what releases every
    transport it opened, and a channel leaked past the run holds a ring buffer
    a real deployment would have handed back.
    """

    def __init__(self, driver: DeviceCallChannelDriver) -> None:
        self._driver = driver
        self._channels: dict[int, DeviceCallChannel] = {}
        self._closed = False

    @property
    def device_ids(self) -> tuple[int, ...]:
        """The device ids that currently have an open channel, in open order."""
        return tuple(self._channels)

    def open_channel(self, device_id: int, name: str) -> DeviceCallChannel:
        """Open the channel registered as `name` for `device_id`.

        Two device ids may share one channel name and still get separate
        channels, because a device id selects a session rather than a transport.
        """
        self._require_open()
        if type(device_id) is not int or device_id < 0:
            raise RealtimeCallError("a device id must be a non-negative integer")
        if device_id in self._channels:
            raise RealtimeCallError(
                f"device id {device_id} already has an open channel; close it first"
            )
        channel = self._driver.open(name)
        self._channels[device_id] = channel
        return channel

    def channel(self, device_id: int) -> DeviceCallChannel:
        """Return the channel open for `device_id`."""
        self._require_open()
        channel = self._channels.get(device_id)
        if channel is None:
            known = ", ".join(str(key) for key in sorted(self._channels)) or "none"
            raise RealtimeCallError(
                f"device id {device_id} has no open channel; open device ids are {known}"
            )
        return channel

    def call(
        self,
        device_id: int,
        function_id: int,
        arguments: bytes = b"",
        *,
        request_id: int = 0,
        ptp_timestamp: int = 0,
    ) -> DeviceCallOutcome:
        """Carry one call to `device_id` and return the response."""
        return self.channel(device_id).call(
            function_id,
            arguments,
            request_id=request_id,
            ptp_timestamp=ptp_timestamp,
        )

    def close(self) -> None:
        """Close every open channel. Closing twice is a no-op."""
        for channel in self._channels.values():
            channel.close()
        self._channels = {}
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RealtimeCallError("the realtime session is closed")

    def __enter__(self) -> RealtimeSession:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
