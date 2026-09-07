# Tensor-network numerics

This package owns tensor-network numerical data and algorithms. It does not
select devices, schedule distributed work, manage jobs, or produce runtime
evidence.

- Start in `models.py` for tensor nodes, contraction plans, compiled schedules,
  slicing plans, and local expectation plans.
- Start in `state.py` for the local tensor-network state, cached contractions,
  observables, amplitudes, sampling, and numerical diagnostics.
- Start in `stages.py` for pair contraction, pair pullback, compiled-stage
  execution, high-rank fallback, and compensated accumulation.
- Start in `path_search.py` for greedy, multistart, tree-reconfiguration, beam,
  and bounded-optimal contraction-order search.
- Start in `contraction.py` for slicing-plan construction, contraction profiles,
  and sliced contraction execution.
- Start in `local.py` for Circuit/IR adaptation, local plan construction, and
  local numerical execution.
- Start in `observables.py` for Pauli/Hamiltonian plans, MPO compression, and
  batched observable contraction.
- Start in `entrypoints.py` for the stable tensor-network wrappers, amplitude
  projections, and Circuit/IR input adaptation.
- Import the owning submodule directly; this package does not re-export a
  tensor-network facade.
- Run `python -m pytest tests/test_tensor_network.py -q` after a typical local
  change.
