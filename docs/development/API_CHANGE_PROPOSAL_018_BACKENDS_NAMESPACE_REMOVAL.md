# API Change Proposal 018: Remove the ambiguous backends facade

## Status

Approved before the first public release. The API owner explicitly authorized
removal on 2026-09-09; no compatibility namespace is retained.

## Decision

Remove `flagquantum.backends`. It contained no implementation and combined
Runtime execution, device resolution, MPS simulation, and tensor-network
simulation under one ambiguous name.

The authoritative expert entry points are now:

- `flagquantum.runtime.run_native`, `run_target`, `run_noisy_mps`,
  `run_noisy_statevector`, and `resolve_device` for Runtime-owned execution
  and device policy;
- `flagquantum.simulation.mps.run_mps` for MPS simulation;
- `flagquantum.simulation.tensor_network` for tensor-network execution,
  amplitudes, and expectations;
- `flagquantum.run` for the normal product workflow.

The future extension discovery mechanism must use explicit domain kinds such as
compiler, provider, and execution backend. It must not recreate a cross-domain
`backends` facade.

## Acceptance

- the removed package is not importable;
- repository code, examples, benchmarks, capability manifests, and maintained
  documentation use the authoritative owner modules;
- Runtime and Simulation focused tests pass;
- architecture and repository-hygiene checks pass.
