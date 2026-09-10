# Phase 27 bounded topology-legalization evidence

Date: 2026-09-10

## Accepted profile

Compiler routes the existing Core `CircuitIR` against one explicit undirected
`CouplingMap`. Restore-after-each-gate and persistent-layout strategies are
supported; `auto` uses the existing deterministic cost comparison. A fresh
coupling-map instance isolates routing output from mutable path-cache history.

The routed artifact must place every two-wire instruction on a coupling edge,
restore the identity logical output layout, report a valid inserted-SWAP count,
and remain within a default 256-added-operation budget. Circuit parameter
objects remain attached by reference.

The combined target-legality order is topology routing, evidenced native-gate
legalization, backend lowering validation, and final target-resource matching.
Inserted SWAPs use the exact three-CX decomposition when SWAP is not native.

## Verification summary

| Gate | Result |
| --- | --- |
| Nonlocal CX routes entirely onto line-topology edges | pass |
| Routed and source statevectors agree | pass |
| Parameter objects and gradients survive routing | pass |
| Auto strategy and serialized routed CircuitIR are deterministic | pass |
| Final logical layout is restored to identity | pass |
| Nonlocal channels fail the topology postcondition | pass |
| Routing growth above policy fails closed | pass |
| Paths requiring unallocated physical ancillas fail closed | pass |
| Inserted SWAPs decompose to native CX before backend checks | pass |
| Final resource checks observe routing and decomposition growth | pass |
| Phase 27 topology tests | 5 passed |
| Hybrid compiler and private-contract focused suite | 188 passed |

## Claim boundary

The coupling map is an explicit caller input bound to a target snapshot ID.
Target Capabilities v1 does not currently provide topology facts, so this phase
does not claim snapshot-backed topology provenance. Directed gates, placement,
physical ancillas, calibration or fidelity-aware routing, crosstalk, timing,
parallel scheduling, target emission, provider execution, TargetIR,
MLIR/LLVM/QIR, public APIs, default-path changes, and performance claims remain
outside the accepted profile.
