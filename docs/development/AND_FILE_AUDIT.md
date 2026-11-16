# And and MultiAnd whole-file audit

The source files, original tests, and 30-cell notebook are pinned to
Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`.
Compiler owns the shared four-T And, measured erasure, and literal
MultiAnd ladder. Simulation owns classical values and separately
the five-axis conjunction tensor. Drawer owns the musical score and
the classical dependency graph.

Forward controls are ordered before prefix outputs. The source
signature exposes the middle prefixes as `junk` and the final prefix
as `target`. These are logical outputs, with the adjoint consuming
all of them before releasing their wires. Native examples use
`circuit.append(op)`, caller-owned quantum wires, and explicit
classical feedback slots.

## Notebook and test file acceptance

Q-0544 closes with `examples/notebooks/register_and.ipynb`. All 30
source cell positions and 17 code cells are retained. Both signatures
and call graphs, Clifford+T compute and measured erase, six mixed
controls, state/effect contraction, and both classical examples
execute through existing native owners. The independent native
kernel blocks reference imports and produced six SVGs. The source
notebook also ran independently.

Q-0546 closes with the source test dispositions and corresponding
native evidence:

| Original tests | Native evidence |
| --- | --- |
| And/MultiAnd example and symbolic resource tests | Ordered signatures and symbolic and concrete call graphs; native symbolic T formula and concrete legacy counts |
| Truth tables, unknown target, bad adjoint and input validation | Existing value callbacks and physical predicate admission, plus state/effect contraction compared on all eight notebook inputs |
| Inverse, MultiAnd adjoint and AndIdentity | Existing all-small-history erasure and the complete complex test state through both AndIdentity branches |
| Random classical simulation | All 20 cases from the original seeded fixture, including every prefix output |
| Musical score and connected diagram | Four polarity glyph sets, side and lane geometry, existing MultiAnd score comparisons, and notebook graph |
| T complexity | All four And polarities and three original MultiAnd lengths/polarity cases compared with the original legacy T and Clifford totals |

The unchanged source suite reports 36 passes and three skips. Its
atomic And has no composite decomposition, call-count body, or
decomposition-dependent type check. Native tests compare its
physical Clifford+T stream and its atomic tensor separately.
The mixed six-control example matches both native primitive streams
gate for gate, including every feedback slot. It emits 20 T/T†
during compute and five measurements during erase. All 32 forced
measurement histories reproduce the original complex input with
all five prefix wires zero, at the expected per-branch probability
of 1/32. The source model separately declares five And calls and
six polarity X calls.

Q-0545 remains open for source per-control symbolic metadata, including
the symbolic Cirq diagram labels. Symbolic width and call-count parity
are covered. The physical compiler admits concrete binary activation
values and uses the existing measured adjoint, with no added symbolic
physical gate contract.
