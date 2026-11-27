# Training lifecycle

Owns fq.Module integration, the training loop and training compilation orchestration. Checkpoint persistence is owned by `../checkpointing.py`, while numerical gradients stay with the executors and Simulation. The two existing Compiler capture/compilation entries remain the only specifically approved frontend exceptions.

Allowed dependencies: Runtime policy/results/executors, Core contracts, Torch and approved Compiler entries.

Published entry points: fq.Module and the existing training APIs. Historical import paths are forwarding facades;
the implementation in this directory is authoritative. See the
[relocation ledger](../../../docs/development/INDEPENDENT_STRUCTURE_MIGRATION.md)
for compatibility ownership and removal gates.

## Ten-minute contributor path

1. Read the implementation matching the behavior under change and its existing test.
2. Make one focused change inside this domain, keeping published contracts unchanged.
3. From the repository root, run:

   ```sh
   python -m pytest tests/unit/test_module_runtime_binding.py tests/unit/test_statevector_training.py -q
   python tools/architecture/check_architecture.py
   ```

CPU tests establish behavior and fixture conformance; physical hardware and
distributed capacity require their separate evidence gates.
