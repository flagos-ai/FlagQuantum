# devices/__init__.py
"""Compatibility access to the legacy distributed quantum device."""

import logging

from ..runtime.backends.statevector.legacy_device import DistributedQuantumDevice

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["DistributedQuantumDevice"]
