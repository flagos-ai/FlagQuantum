# Single-factorization PREPARE audit

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/sf`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0399 | `__init__.py` | `ce76a8d10a5909daeb7b1113c8cc7f26cc31b410` |
| Q-0400 | `prepare.py` | `1ad5861baadd1e1261f98cf74c052a27e2e2803f` |
| Q-0401 | `prepare_test.py` | `859f4edabdeedb5feffad270cf62f207b5ffc187` |

The package initializer contains only licensing text. The existing
single-factorization algorithm owners retain the inner and outer
binding factories.

## Narrow source declarations

The source outer signature has the factor address `l` and three flags
`succ_l`, `l_ne_zero`, and `rot_aa`. The source inner signature has
`l`, two spatial indices, and `succ_pq`. All ports are through.
An absent index is omitted when its width is zero. Both resource
models use the shared signatures for their local widths.
The assertions compare the original order, shape, side and dtype in
both directions, including the original inner blocking factors of four and eight in the small teaching fixture.

At the original 50-factor fixtures, the outer declaration estimates
46 Toffoli and 40 And forward, and 15 Toffoli and 40 And adjoint.
The inner declaration estimates 682 Toffoli and 23 And forward,
and 94 Toffoli and 23 And adjoint. Both fixtures have a 60-qubit
modeled peak in either direction. Those estimates retain the QROAM
eraser asymmetry. The tests also check each immediate child budget
and multiplicity.

Both pinned classes declare cost graphs without a physical
decomposition. The source outer automatic blocking still refuses
a small table with fewer rows than target bits. The narrow source
signature remains available when the resource query cannot run.

## Executable dependencies

The already accepted Appendix B8/B11 implementations attach through
`circuit.append(op)`. They expose the loaded alias information and
coherent sign registers required by the executable bindings, in
addition to the modeled source ports. Their matched inverses consume
those outputs and return temporary work to zero. Existing full
complex-amplitude and measurement-history checks are part of the
concentrated regression. They cover both directions and one-element
axis folding.

The four-row outer fixture emits 45 T/T† and one measurement using two
work qubits. The three-orbital dual-axis inner fixture emits 219 T/T†
and 25 measurements using seven work qubits. Those emitted values carry
no claim of equality to a source gate stream.

The original tests returned six passes and six skips with OpenFermion
1.6.1. These results and the native declaration and execution evidence
close Q-0399 through Q-0401 in the original source scope.

The concentrated native regression passed 46 cases without skips.
Strict typing, formatting, ownership, architecture, IR contract, and
public API checks passed.
