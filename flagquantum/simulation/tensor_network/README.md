# Tensor-network numerics

This package owns tensor-network numerical data and algorithms. It does not
select devices, schedule distributed work, manage jobs, or produce runtime
evidence.

- Start in `models.py` for tensor nodes, contraction plans, compiled schedules,
  slicing plans, and local expectation plans.
- Import the owning submodule directly; this package does not re-export a
  tensor-network facade.
- Run `python -m pytest tests/test_tensor_network.py -q` after a typical local
  change.
