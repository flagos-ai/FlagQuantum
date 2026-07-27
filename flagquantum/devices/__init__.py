# devices/__init__.py
"""Quantum device implementations.

This module provides different quantum device backends:
- DistributedQuantumDevice: Multi-GPU distributed statevector simulator
"""

import logging

from .distributed_device import DistributedQuantumDevice

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["DistributedQuantumDevice"]
