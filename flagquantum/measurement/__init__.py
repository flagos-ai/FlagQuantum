# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

from .measure import *  # noqa: F403

logging.getLogger(__name__).addHandler(logging.NullHandler())
# Note: All exports are controlled by each submodule's __all__
# No need to manually list exports here
