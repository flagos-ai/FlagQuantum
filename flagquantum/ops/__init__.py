# ops/__init__.py
"""Quantum gate operations.

This module provides quantum gate matrices, functional operations,
operator classes, invertible modules, and gate registration utilities.

Examples
--------
Import and use gates:

>>> from flagquantum.ops import rx, cx, H
>>> from flagquantum.devices import DistributedQuantumDevice
>>>
>>> device = DistributedQuantumDevice(n_wires=2, bsz=4)
>>> device.rx(wires=[0], params=0.5)
>>> device.cx(wires=[0, 1])

Using operator classes:

>>> from flagquantum.ops import RX, CX
>>> rx_gate = RX(wires=[0], params=torch.tensor(0.5))
>>> cx_gate = CX(wires=[0, 1])
>>> rx_gate(device)
>>> cx_gate(device)

Using invertible module with noise:

>>> from flagquantum.ops import InvertibleUnitary, make_noisy_layer
>>> gates = [RX(wires=[0]), CX(wires=[0, 1])]
>>> noisy_layer = make_noisy_layer(gates, error_rate_1q=0.01)
>>> noisy_layer(device, inp)

Register custom gates:

>>> from flagquantum.ops import register_gate
>>> my_gate = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
>>> register_gate("my_gate", my_gate)

Precision control:

>>> from flagquantum.ops import set_global_precision
>>> import torch
>>> set_global_precision(torch.complex128)  # Use double precision
"""

import logging

# Import all from submodules
from .functional import *  # noqa: F403
from .invertible import *  # noqa: F403
from .matrices import *  # noqa: F403
from .operator import *  # noqa: F403
from .registry import *  # noqa: F403

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Note: All exports are controlled by each submodule's __all__
# No need to manually list exports here
