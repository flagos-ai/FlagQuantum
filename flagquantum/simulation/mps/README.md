# MPS numerics

This package owns matrix-product-state numerical data and algorithms. It does
not select devices, schedule distributed work, manage jobs, or produce runtime
evidence.

- Start in `models.py` for MPS configuration, compiled local schedules, and
  numerical result records.
- Start in `state.py` for the batched MPS representation, local state
  operations, observables, and truncation diagnostics.
- Import the owning submodule directly; this package does not re-export an MPS
  facade.
- Run `python -m pytest tests/test_mps.py -q` after a typical local change.
