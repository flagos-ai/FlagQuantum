"""Historical numerical import facade."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.tensor_network.jax.models import (
    JAXTensorNetworkNode as JAXTensorNetworkNode,
)

_sys.modules[__name__] = _import_module(
    "flagquantum.simulation.tensor_network.jax.models"
)
