# Montgomery view graph import

The immutable reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

## Interface and ownership

The private Ecosystem graph importer immediately converts live graphs into owned
Program records. Split and Join already delegate to Compiler word rebindings.
Their bit order is MSB first. No work, allocation, measurement or numeric
Montgomery conversion occurs in these views.

The original serializer omits the modulus from QMontgomeryUInt. Distinct source
Split/Join objects with width 3 and modulus None or 3 therefore have identical
owned declarations. The importer previously rejected this actual controlled
ECWindowAddR dependency before reaching the existing native rebinding owner.

Identical declarations may now share the same owned definition only for exact
Split/Join classes with exact Montgomery dtype. Arbitrary collisions, including
Allocate/Free with the same serialized types, still refuse. Foreign subclasses
and other dtype families retain the ambiguity guard.

## Evidence and remaining scope

The baseline reproducer fails both graph directions at the ambiguity guard.
Both source dtype methods produce identical MSB bit maps and inverse joins for
all eight raw storage patterns, including encodings outside the residue range.
The resulting native forward and adjoint graphs preserve reversed physical
bindings, both nominal owned dtypes, the coherent full 128-amplitude state with
a spectator, JSON and the existing `circuit.append(op)` entry. They emit zero
primitive gates, lifetime events or feedback slots. Identity views have no
independent nonzero resource estimate. Existing borrowed/clean graph regressions
remain accepted independently.

The actual controlled ECWindowAddR graph now passes this collision and reaches
`No native factory for qualtran.bloqs.data_loading.qroam_clean.QROAMClean`.
Its larger foreign graph execution remains open. Receipts are
`g2-montgomery-rebinding-baseline-20261121.log`,
`g2-montgomery-rebinding-regression-20261121.log` and
`g2-montgomery-rebinding-followup-20261121.json` in the reference directory.
