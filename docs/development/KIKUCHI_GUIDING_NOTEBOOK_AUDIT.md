# Kikuchi guiding-state notebook audit

The complete Qualtran v0.7.0 notebook is frozen at revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`ca6e8c12a2c0f2e387609045996675e2738b5205`. Copyright 2024 Google LLC,
adapted under Apache-2.0.

`examples/notebooks/kikuchi_guiding_state.ipynb` preserves all 21 original cell
roles and adds one native evidence cell. Original Markdown teaches scope states,
tensor-product guiding states, precision, amplitude bounds and signal flags.
Code uses the domain-owned five factory counterparts and shared signature and
call-graph drawing. Source `SimpleGuidingState` and
`SparseStatePreparationViaRotations` map to native `ScopeGuidingState` and
`SparseRotationPreparation`. Those two semantic name mappings are explicit in
both graph comparison and notebook text.

## Complete-cell verification

The untouched original and native notebook execute every code cell. Native
execution rejects imports of Qualtran, Cirq and OpenFermion. All five factory
instances match source signatures, sides, constraints, concrete and symbolic
seven-category QEC models. Both source call graphs match after the two declared
native name mappings. Five signatures and two call graphs render through the
existing drawer. All seven images receive visual inspection.

The added evidence cell reports concrete and symbolic QEC models. Its concrete
root has 132 Toffoli, 192 CSwap, 352 And, 4173 Clifford and 352 measurements.
These are model estimates. Primitive emission and all small measurement branch
verification belong to the executable coherent-uncompute dependency documented
in `KIKUCHI_GUIDING_STATE_FILE_AUDIT.md`.

The batch regression passes 962 tests with three visible warnings. The shared
notebook test and complete extracted code pass strict mypy, Black and Ruff.
Team scope, architecture, IR and API checks pass.

## Scope disposition

Q-0610 closes as a complete declaration/model notebook. Q-0611 remains active
at the frozen scope root's missing decomposition/phase-gradient port, source
sorting's missing physical decomposition. The current typed root admits its
bounded permutation array under the existing Core contract; see the later
measured-child context in KIKUCHI_GUIDING_STATE_FILE_AUDIT.md. Full-root physical amplitudes and cleanup are unverified. Q-0612
retains its original skips/count-consistency and serialization xfails. Q-0613,
the separate tutorial notebook, is the next whole file.

The tutorial follow-up now retains literal symbolic/concrete word roots and
the original one-level expansion, with twelve/twenty-one calls respectively.
Concrete duplicate detection supplies its existing source-faithful body;
only preparation and sorting lack original physical bodies. This does not
close the separate tutorial: both resource graph views, complete native cell
execution and final six-figure inspection remain required.
