# Hybrid compilation Phase 0 inventory

Updated: 2026-09-09
Status: **inventory and private contract complete; implementation entry blocked
by the assigned Compiler worktree gate**

## Decision

Proceed with a Python-first, FlagQuantum-owned hybrid compilation vertical slice.
Do not restore the former private compiler tree wholesale. Reuse its validated
semantic patterns selectively inside the current authoritative Compiler domain.

The first scenario supports classical tensor values controlling loops and gate
selection before an analytic scalar expectation. Catalyst, C++, native QIR,
mid-circuit measurement feedback, noise, finite shots, and distributed execution
are excluded.

## Current authoritative path

```text
Circuit / CircuitIR
  -> flagquantum.compiler
  -> CircuitIR
  -> Runtime plan and execution
  -> local Simulation
```

The active compiler is circuit-level. Its stable transformation contract accepts
and returns Core-owned `CircuitIR`. The installed compiler extension proposal has
the same `CircuitIR -> CircuitIR` boundary. Hybrid work must not overload this
contract with a different artifact type.

## Historical implementation located

The retained workspace copy at:

```text
/srv/quantum-demo/FlagQuantum/FlagQuantum/flagquantum/_compiler
```

contains a private compiler research implementation. Repository history records
its removal in commit:

```text
fb3da2ee compiler: remove disconnected private compiler tree
```

The removal affected 197 files and approximately 28.8K lines. The stated reason
was architectural disconnection, not failed semantic tests.

## Historical capability classification

### Reusable semantic patterns

- immutable `IRType`, `ValueId`, and `ValueRef`;
- immutable operation attributes and deterministic canonical encoding;
- `Block`, `Region`, and module containment;
- qualified operation schemas;
- fail-closed structural verification;
- linear qubit-value verification;
- def-use and qubit-lifetime analyses;
- explicit pass preconditions and preserved properties;
- deterministic program, target, and artifact identities;
- target capability checks and sealed artifact integrity.

### Reusable only after simplification

- pass manager: retain only behavior required by the first profile;
- TargetIR: defer until the vertical slice needs physical targeting;
- executable artifact: align with current artifact ownership before migration;
- cache: retain structure/value identity separation, remove unused target and
  provider dimensions;
- text emitters: not required by the first CPU vertical slice.

### Do not restore

- provider discovery, submission, queue, receipt, and credential machinery;
- deployment bridge stages and shadow harnesses;
- historical approval packets as implementation dependencies;
- public compatibility wrappers for removed private types;
- duplicate runtime adapters or execution result models;
- performance baselines that no longer describe the active code path.

## What was actually implemented

The historical code implemented a private static `QuantumIR`, not dynamic
`ProgramIR`. It included typed values, generic nested regions, operation schemas,
verification, static circuit import/export, canonicalization, decomposition,
routing, target legalization, artifact sealing, and provider-neutral runtime
contracts.

The following were not implemented:

- dynamic `ProgramIR` functions and calls;
- registered `if`, `for`, and `while` semantics;
- control-flow lowering and branch joins for quantum values;
- measurement-produced values controlling later operations;
- hybrid automatic differentiation;
- Python AutoGraph-style capture;
- QIR, timing, or pulse lowering;
- integration with the default `fq.run`/`fq.plan` path.

Generic `Region` storage is infrastructure, not evidence of control-flow
semantics.

## Verification performed

The following retained tests were executed against the historical workspace:

```text
test_module_model.py
test_types_and_values.py
test_operation_schema.py
test_verifier_negative.py
test_circuit_ir_round_trip.py
test_phase2_static_canonicalization.py
test_phase3_target_legalization.py
test_phase3_executable_artifact.py
```

Result:

```text
84 passed in 4.53s
```

This proves that the selected historical components are executable research
assets. It does not prove current-vNext integration, dynamic control flow,
hybrid gradients, production performance, or public stability.

## Current repository blockers

The integration worktree is on the correct authoritative branch. The earlier
compiler-extension/public API changes have been incorporated upstream; only the
hybrid-plan, inventory, private contract, and matching contract test are pending
in this worktree.

The assigned Compiler worktree is clean, but it is currently checked out on a
phase-specific branch rather than the `team-ownership.toml` Compiler branch.
Repository policy prohibits switching another team's worktree or starting a
second writing session there without reconciliation.

Consequently, Phase 1 code implementation must not begin until:

1. the Compiler worktree is returned by its owner to the assigned Compiler
   branch and synchronized with the resulting integration baseline;
2. team scope preflight passes for the exact Phase 1 files.

## Phase 1 proposed file scope

```text
flagquantum/compiler/hybrid/README.md
flagquantum/compiler/hybrid/types.py
flagquantum/compiler/hybrid/values.py
flagquantum/compiler/hybrid/operations.py
flagquantum/compiler/hybrid/regions.py
flagquantum/compiler/hybrid/schemas.py
flagquantum/compiler/hybrid/verifier.py
tests/hybrid_compiler/test_program_ir.py
tests/hybrid_compiler/test_verifier_negative.py
```

The Phase 1 change is compiler-only and does not execute numerical kernels. It
is classified as compiler infrastructure supporting a future
`single_device_fast_path`; it makes no runtime or scalability claim.

## Phase 0 exit checklist

- [x] Problem and first vertical slice defined
- [x] Current authoritative types identified
- [x] Historical implementation and removal reason identified
- [x] Reuse and subtraction decisions recorded
- [x] Static versus hybrid capability boundary recorded
- [x] Historical test sample executed
- [x] Public API impact constrained to none
- [x] Existing integration worktree changes resolved
- [x] Private Phase 1 semantic contract approved conditionally
- [x] Machine-readable contract and contract test added
- [ ] Assigned Compiler worktree reconciled
- [ ] Phase 1 team-scope preflight passed
