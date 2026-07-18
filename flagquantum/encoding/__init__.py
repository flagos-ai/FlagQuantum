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

# encoding/__init__.py
"""Quantum encoding methods for data embedding.

This module provides various quantum encoding techniques to embed
classical data into quantum states.

Examples
--------
Using GeneralEncoder:

>>> from flagquantum.encoding import GeneralEncoder
>>>
>>> func_list = [
...     {"func": "ry", "wires": [0], "input_idx": 0},
...     {"func": "ry", "wires": [1], "input_idx": 1},
... ]
>>> encoder = GeneralEncoder(func_list)
>>> encoder(device, x)

Using convenience functions:

>>> from flagquantum.encoding import angle_encoder, basis_encoder, amplitude_encoder
>>>
>>> angle_encoder(device, x, wires=[0, 1])
>>> basis_encoder(device, binary_data, wires=[0, 1])
>>> amplitude_encoder(device, amplitudes)

Using factory:

>>> from flagquantum.encoding import create_encoding_circuit
>>> encoder = create_encoding_circuit("angle", n_qubits=4, n_features=4)
"""

import logging

from .encoder import *  # noqa: F403

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Note: All exports are controlled by each submodule's __all__
# No need to manually list exports here
