"""Historical numerical import facade."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.mps.jax.batched import (
    _discarded_weight as _discarded_weight,
)
from flagquantum.simulation.mps.jax.batched import _select_rank as _select_rank
from flagquantum.simulation.mps.jax.batched import (
    jax_mps_apply_one_batched as jax_mps_apply_one_batched,
)
from flagquantum.simulation.mps.jax.batched import (
    jax_mps_apply_two_batched as jax_mps_apply_two_batched,
)
from flagquantum.simulation.mps.jax.batched import (
    jax_mps_split_pair_batched as jax_mps_split_pair_batched,
)

_sys.modules[__name__] = _import_module("flagquantum.simulation.mps.jax.batched")
