# measurement/__init__.py
"""Quantum measurement operations.

This module provides measurement functions for quantum devices.

Examples
--------
Basic measurement:

>>> from flagquantum.measurement import measure_allZ
>>> from flagquantum.devices import DistributedQuantumDevice
>>>
>>> device = DistributedQuantumDevice(n_wires=4, bsz=2)
>>> device.h(wires=[0])
>>> expectations = measure_allZ(device)

With sampling:

>>> outcomes = measure_allZ(device, shots=1024)

With post-selection:

>>> expectations, retained = measure_allZ(device, postselect_cond={0: 1})
"""

import logging

from .measure import measure_allZ, sampler_diff_approx, sampler_nondiff_exact

logging.getLogger(__name__).addHandler(logging.NullHandler())
__all__ = ["measure_allZ", "sampler_diff_approx", "sampler_nondiff_exact"]
