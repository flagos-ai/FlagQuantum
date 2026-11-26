"""Historical numerical import facade."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.statevector.jax.kernels import (
    _jax_apply_cx_permutation as _jax_apply_cx_permutation,
)
from flagquantum.simulation.statevector.jax.kernels import (
    jax_accumulate_all_to_all_statevector_delta as jax_accumulate_all_to_all_statevector_delta,
)
from flagquantum.simulation.statevector.jax.kernels import (
    jax_apply_local_statevector_gate as jax_apply_local_statevector_gate,
)
from flagquantum.simulation.statevector.jax.kernels import (
    jax_apply_matrix_to_batched_local_state as jax_apply_matrix_to_batched_local_state,
)
from flagquantum.simulation.statevector.jax.kernels import (
    jax_basis_indices_for_qubits as jax_basis_indices_for_qubits,
)
from flagquantum.simulation.statevector.jax.kernels import (
    jax_combine_pair_exchanged_statevector as jax_combine_pair_exchanged_statevector,
)
from flagquantum.simulation.statevector.jax.kernels import (
    jax_gate_basis_in_for_delta_and_local_input as jax_gate_basis_in_for_delta_and_local_input,
)
from flagquantum.simulation.statevector.jax.kernels import (
    jax_initial_statevector_shard as jax_initial_statevector_shard,
)
from flagquantum.simulation.statevector.jax.kernels import (
    jax_local_positions_for_gate_input as jax_local_positions_for_gate_input,
)
from flagquantum.simulation.statevector.jax.kernels import (
    jax_rank_mask_for_touched_delta as jax_rank_mask_for_touched_delta,
)
from flagquantum.simulation.statevector.jax.kernels import (
    jax_sharded_statevector_loss as jax_sharded_statevector_loss,
)
from flagquantum.simulation.statevector.jax.kernels import (
    jax_sharded_statevector_rank_loss as jax_sharded_statevector_rank_loss,
)
from flagquantum.simulation.statevector.jax.kernels import (
    jax_statevector_from_circuit as jax_statevector_from_circuit,
)

_sys.modules[__name__] = _import_module(
    "flagquantum.simulation.statevector.jax.kernels"
)
