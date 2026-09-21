# Qiskit seeded differential conformance

## Purpose

The Qiskit adapter must preserve executable quantum semantics in both
directions. Individual gate tests establish local mappings, but they do not
exercise interactions among wire order, parameter order, multi-wire gates, and
custom unitary basis conventions. The seeded differential corpus adds that
compositional evidence without introducing Qiskit objects into FlagQuantum IR
or runtime code.

## Corpus

`run_qiskit_conformance()` executes six deterministic programs identified by
their seeds: `731`, `946`, `1212`, `1597`, `2018`, and `2371`. Each program:

- uses three to five wires;
- prepares every wire with a nontrivial rotation;
- combines twelve fixed and parameterized one-, two-, and three-wire gates;
- samples ordered wire tuples, including non-contiguous and reversed layouts;
- ends with an asymmetric two- or three-wire custom unitary; and
- can be reproduced from the seed embedded in its case name.

The custom unitary generator uses a seeded permutation with independent complex
phases. It is exactly unitary by construction and requires no numerical matrix
factorization. The independent Qiskit builder converts local matrix indices
with an integer bit-reversal permutation instead of calling the adapter's
conversion helper.

## Differential oracles

For every seeded program, conformance compares three paths in complex128:

1. native FlagQuantum execution against an independently constructed Qiskit
   circuit;
2. native FlagQuantum execution against the FlagQuantum-to-Qiskit export; and
3. direct Qiskit execution against execution after Qiskit-to-FlagQuantum import.

The maximum absolute error across all three comparisons is recorded in the
existing machine-readable conformance result. Export/import fingerprints also
have to match. Qiskit labels and adapter-source annotations are excluded from
that fingerprint because they do not affect executable circuit semantics.

## Boundary

This corpus certifies only the declared Qiskit adapter surface on the pinned
Qiskit 2.0.x and 2.5.x CI lanes. Fixed seeds make failures reproducible; they do
not prove correctness for arbitrary circuits, provider execution, hardware,
noise models, or unsupported control flow.
