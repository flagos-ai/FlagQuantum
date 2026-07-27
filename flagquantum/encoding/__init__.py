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

from .encoder import (
    GeneralEncoder,
    amplitude_encoder,
    angle_encoder,
    basis_encoder,
    create_encoding_circuit,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "GeneralEncoder",
    "amplitude_encoder",
    "angle_encoder",
    "basis_encoder",
    "create_encoding_circuit",
]
