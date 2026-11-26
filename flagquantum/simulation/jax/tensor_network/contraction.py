"""Historical numerical import facade."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.tensor_network.jax.contraction import (
    _choose_greedy_pair as _choose_greedy_pair,
)
from flagquantum.simulation.tensor_network.jax.contraction import (
    _estimate_pair as _estimate_pair,
)
from flagquantum.simulation.tensor_network.jax.contraction import (
    _label_counts as _label_counts,
)
from flagquantum.simulation.tensor_network.jax.contraction import (
    _label_dimensions as _label_dimensions,
)
from flagquantum.simulation.tensor_network.jax.contraction import (
    _pair_output_labels as _pair_output_labels,
)
from flagquantum.simulation.tensor_network.jax.contraction import (
    contract_assigned_slices as contract_assigned_slices,
)
from flagquantum.simulation.tensor_network.jax.contraction import (
    contract_nodes_greedy as contract_nodes_greedy,
)
from flagquantum.simulation.tensor_network.jax.contraction import (
    slice_nodes as slice_nodes,
)
from flagquantum.simulation.tensor_network.jax.contraction import (
    zero_for_output as zero_for_output,
)

_sys.modules[__name__] = _import_module(
    "flagquantum.simulation.tensor_network.jax.contraction"
)
