# Optimization workflows

Owns QAOA, VQE/ADAPT/layerwise workflows, QUBO objectives, ground-state composition and its research-provider dispatch. The objective and workflow live here; reusable optimizer update rules live in the sibling optimizers package. Do not implement device adaptation, numerical state evolution or Compiler lowering.

Allowed dependencies: Core operator models, algorithm primitives and established fq execution interfaces.

Published entry points: Existing Hamiltonian/VQE/QAOA and ground-state APIs. Historical import paths are forwarding facades;
the implementation in this directory is authoritative. See the
[relocation ledger](../../../docs/development/INDEPENDENT_STRUCTURE_MIGRATION.md)
for compatibility ownership and removal gates.

## Ten-minute contributor path

1. Read the implementation matching the behavior under change and its existing test.
2. Make one focused change inside this domain, keeping published contracts unchanged.
3. From the repository root, run:

   ```sh
   python -m pytest tests/team/services/test_ground_state_vqe.py -q
   python tools/architecture/check_architecture.py
   ```

CPU tests establish behavior and fixture conformance; physical hardware and
distributed capacity require their separate evidence gates.
