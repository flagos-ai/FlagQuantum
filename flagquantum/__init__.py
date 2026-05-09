# flagquantum/__init__.py
"""FlagQuantum - A distributed quantum computing framework.

FlagQuantum is a PyTorch-based quantum computing framework with built-in
support for distributed simulation across multiple GPUs.

Examples
--------
>>> import flagquantum as fq
>>> device = fq.DistributedQuantumDevice(n_wires=4, bsz=2)
>>> fq.h(device, wires=[0])
>>> fq.rx(device, wires=[1], params=0.5)
>>> fq.cx(device, wires=[0, 1])
>>> results = fq.measure_allZ(device)
"""

import logging

# Also expose submodules for advanced users
from . import devices, encoding, measurement, ops, utils

# ============================================================================
# Import from submodules (__all__ is automatically aggregated)
# ============================================================================
from .devices import *  # noqa: F403
from .encoding import *  # noqa: F403
from .measurement import *  # noqa: F403
from .ops import *  # noqa: F403
from .version import __version__, get_version

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
# ============================================================================
# Package info
# ============================================================================

__author__ = "FlagQuantum Team"
__license__ = "Apache-2.0"


def info() -> dict:
    """Get package information."""
    return {
        "name": "flagquantum",
        "version": __version__,
        "author": __author__,
        "license": __license__,
    }


def hello() -> None:
    """Print welcome message."""
    print(f"FlagQuantum v{__version__} - Distributed Quantum Computing Framework")


# ============================================================================
# Module exports
# ============================================================================

__all__ = (
    # Version
    ["__version__", "get_version", "info", "hello"]
    +
    # Submodules
    ["devices", "ops", "encoding", "measurement", "utils"]
    +
    # Devices exports
    (devices.__all__ if hasattr(devices, "__all__") else [])
    +
    # Ops exports
    (ops.__all__ if hasattr(ops, "__all__") else [])
    +
    # Encoding exports
    (encoding.__all__ if hasattr(encoding, "__all__") else [])
    +
    # Measurement exports
    (measurement.__all__ if hasattr(measurement, "__all__") else [])
)
