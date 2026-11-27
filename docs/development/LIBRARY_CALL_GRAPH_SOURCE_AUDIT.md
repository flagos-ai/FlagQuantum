# Complete library graph across source SDK profiles

The locked source is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The existing owner is
`tools/development/library_call_graph.py`, using the Compiler's shared symbolic
union traversal. Its callers supply declarations, generalizers and terminal
rules. The comparison uses all 393 original examples and replays the same
constructed roots through the source and native consumers, retaining callable
and external-object identities.

| Source profile | Canonical controlled Y | Complete source/native nodes | Complete source/native edges |
|---|---|---:|---:|
| Cirq 1.6.1 | `cirq_gate_to_bloq` produces atomic `CYGate`. The source generalizer omits its child calls; its explicit example root remains isolated. | 3692 | 8543 |
| Cirq 1.7.0 | The same converter wraps `CYPowGate` as `CirqGateAsBloq`. This wrapper survives source generalization. | 3693 | 8550 |

The class census changes by one `CirqGateAsBloq(CYPowGate)` node. Its six
incoming edges come from `SelectedMajoranaFermion`: three selection sizes,
each with no control or one control, with weights 4, 4, 6, 6, 8 and 8. Its one
outgoing edge has weight two and targets the source-generalized `XPowGate`.
The original positive and negative half-X rotations merge after angle
generalization; the intervening CZ is omitted by the original Clifford rule.
This explains all seven additional edges. The source conversion and
generalizer order are retained, including their model-count consequences.

The prior native test required 3693/8550 before performing any graph comparison,
so it rejected the supported 1.6 profile prematurely. The repair retains that
exact census and checks the complete CY subgraph when applicable. The 1.6
alternative requires the actual canonical `CYGate`, its isolated-root
ownership, and the independently observed 3692/8543 census. Both alternatives
continue through every original node and weighted-edge comparison, native
acyclicity check, and the unchanged upstream acceptance function with the
native graph provider. The upstream function itself checks acyclicity and
contains no fixed node or edge census.

Two diagnostic mistakes were resolved before acceptance. A forced generic CY
wrapper bypassed the standard source converter in 1.6. The canonical atomic
CY also remains as an explicit library root even when generalized away as a
child. The final checks follow the real converter and preserve that root.
Earlier failing diagnostic runs remain recorded; they are not successful
acceptance evidence.

Actual optional SDK imports and dynamic source-module loading use the shared
reference dependency boundary. Mandatory runs retain the original missing
dependency error. Both complete six-test owner suites pass, including the
unchanged original whole-library body. Twelve SDK absence scenarios and two
native consumers are checked independently. Seven actual source interfaces
and eleven source files match their original identities and frozen Git blobs.
Strict typing covers the complete owner file.

The graph contains declared model calls. Physical primitive inventories and
independent historical/modern resource estimates retain their existing
evidence. Production, Compiler and executable component owners are unchanged
by this verification repair. Pending static-CY conversion Proposal 127 remains
outside this work; no source defect is waived and G2 remains open.

Receipts use the external reference workspace prefix
`g2-library-profile-reference-`, dated November 27, 2026. Independent diagnostic
receipts use `g2-library-profile-diagnostic-16/17-20261127` and
`g2-library-cy-edges-17-20261127`.
