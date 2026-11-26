# Quantum machine learning

Owns feature encodings, quantum kernels, hybrid models, feature selection, clustering, PCA and association-rule workflows. Generic training lifecycle stays in Runtime; state simulation stays in Simulation.

Allowed dependencies: Core models, algorithm workflows and established Torch/Runtime interfaces.

Published entry points: Existing quantum-kernel, encoding and model exports. Historical import paths are forwarding facades;
the implementation in this directory is authoritative. See the
[relocation ledger](../../../docs/development/INDEPENDENT_STRUCTURE_MIGRATION.md)
for compatibility ownership and removal gates.

## Ten-minute contributor path

1. Read the implementation matching the behavior under change and its existing test.
2. Make one focused change inside this domain, keeping published contracts unchanged.
3. From the repository root, run:

   ```sh
   python -m pytest tests/unit/test_algorithms_data_encoding.py -q
   python tools/architecture/check_architecture.py
   ```

CPU tests establish behavior and fixture conformance; physical hardware and
distributed capacity require their separate evidence gates.
