"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.realtime.channels import (
    DEFAULT_SLOT_SIZE as DEFAULT_SLOT_SIZE,
)
from flagquantum.compute.realtime.channels import (
    ChannelFactory as ChannelFactory,
)
from flagquantum.compute.realtime.channels import (
    DeviceCallChannel as DeviceCallChannel,
)
from flagquantum.compute.realtime.channels import (
    DeviceCallChannelDriver as DeviceCallChannelDriver,
)
from flagquantum.compute.realtime.channels import (
    DeviceCallOutcome as DeviceCallOutcome,
)
from flagquantum.compute.realtime.channels import (
    LoopbackTransport as LoopbackTransport,
)
from flagquantum.compute.realtime.channels import (
    RingTransport as RingTransport,
)
from flagquantum.compute.realtime.channels import __all__ as __all__
from flagquantum.compute.realtime.channels import (
    loopback_channel as loopback_channel,
)

_implementation = _import_module("flagquantum.compute.realtime.channels")
_sys.modules[__name__] = _implementation
