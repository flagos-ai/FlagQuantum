# Dynamics numerics

Owns Lindblad generators, physical program numerical assembly and actions, problem validation, numerical integrators, density-matrix evolution and its adjoint. This is the numerical dynamics domain; other admitted evolution methods can join when implemented. Runtime evolution prepares requests and execution policy. Do not select resources, submit jobs or import Runtime.

Allowed dependencies: Core operator and measurement semantics plus tensor numerics.

Published entry points: Existing evolution entry points; runtime.evolution composes user workflows. Historical import paths are forwarding facades;
the implementation in this directory is authoritative. See the
[relocation ledger](../../../docs/development/INDEPENDENT_STRUCTURE_MIGRATION.md)
for compatibility ownership and removal gates.

## Ten-minute contributor path

1. Read the implementation matching the behavior under change and its existing test.
2. Make one focused change inside this domain, keeping published contracts unchanged.
3. From the repository root, run:

   ```sh
   python -m pytest tests/unit/test_structure_import_compatibility.py -q
   python tools/architecture/check_architecture.py
   ```

CPU tests establish behavior and fixture conformance; physical hardware and
distributed capacity require their separate evidence gates.

## Physical program evolution

`rydberg.py` assembles the numerical Hamiltonian from verified Core physical
records. `rydberg_presence.py` applies the existing presence segments and
continuations. `ising_braid.py` executes verified F/R actions and sampled
continuations. Core keeps program semantics and physical records; Runtime owns
planning, result assembly and trajectory lifecycle. These numerical modules
do not select resources or admit unsupported programs.

For a small change, begin with `tests/unit/test_rydberg_presence_actions.py` or
`tests/unit/test_ising_braid_actions.py`, then run the corresponding Runtime
scenario tests. These bounded CPU tests prove numerical and orchestration
semantics, without establishing hardware support or scalability.

Run `python -m examples.simulation.ising_braid_execution` for a bounded CPU
example using the public planning and execution path. Change an ordered source
action and inspect the corresponding probabilities before modifying a kernel.

## Problem normalization

`problem.py` owns request, matrix, physical-state and operator normalization.
Import private normalization helpers directly from that module. `lindblad.py`
consumes the helpers it actually uses and owns evolution composition; it does
not re-export unused private helpers for tests. Its seven declared public
exports retain their historical API and serialization identities.

Run `tests/unit/test_lindblad_matrix_free_hamiltonian.py`,
`tests/unit/test_lindblad_vectorized_generator.py`,
`tests/unit/test_lindblad_batching_and_gradients.py` and
`tests/unit/test_lindblad_adjoint_gradient.py` when changing normalization.
These scenarios preserve the independent dense, stencil and adjoint references.
