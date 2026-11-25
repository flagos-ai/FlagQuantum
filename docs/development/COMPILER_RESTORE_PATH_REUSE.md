# Layout restoration path traversal

The native SABRE planner prices a complete route, including the SWAPs needed to
restore the logical output layout. Restoration is repeated while comparing
routing trials and initial placements, so its classical planning cost affects
ordinary compilation even though it emits no additional quantum work.

`compiler/topology/sabre.py:restore_swaps` fixes one leaf of a spanning forest at
a time. Previously it searched for the token-to-leaf path again after each
SWAP. The active tree does not change during that walk: only token positions
change. The implementation now finds the path once and walks its consecutive
edges, removing the leaf only after its token reaches home.

The path in a tree is unique. After each SWAP, its suffix is exactly the path
that another search from the new token position would return. This preserves
the complete ordered SWAP sequence, including its tie choices, and leaves the
initial/final layout contract, routing scores, evidence fields, and public API
unchanged. Unreachable component exchanges still raise an error. Unused device
wires are still excluded from restoration.

A reversed 64-wire line requires 2,016 adjacent SWAPs. Its path-search count
falls from 2,016 to 63, without removing or adding a SWAP. Search work changes
from one breadth-first search per emitted SWAP to at most one per displaced
leaf. The implementation retains one path at a time, with linear temporary
storage; it does not add a cache or another graph representation. Leaf selection
and the existing spanning-forest construction remain unchanged.

This is an internal implementation improvement to FlagQuantum's existing
restoration algorithm. It adds no upstream dependency or algorithm port. The
existing SABRE/Cirq source attributions remain applicable to their respective
parts of the planner.

## Contributor verification

```bash
python -m pytest -q tests/unit/test_compilation_restore_paths.py tests/unit/test_compilation_sabre_layout.py tests/team/compiler/test_routing_conformance.py
python -m examples.target_aware_compilation
```

The focused tests replay tokens for every permutation of five wires on line,
star and ring graphs; pin existing traces; check a longer line's search bound;
and reject invalid permutations and cross-component movement. The existing
routing suites check state, parameter gradients, classical dependencies,
measurement behavior, topology legality, evidence and consumer replacement.

Local timing of this routine is development evidence only. It does not measure
an end-to-end SDK workload, qualify a benchmark host, or establish a ranking
against Qiskit or Cirq. Formal SDKBench performance validation remains separate.
