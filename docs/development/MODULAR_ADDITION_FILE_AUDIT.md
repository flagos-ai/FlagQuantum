# Modular addition source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `mod_addition.ipynb` | `b081513d2c657fb069e0da81f2b4914df531ba2f` | Q-0569 |
| `mod_addition.py` | `0a167e84fe00d5df50488c4b28925ea6114728f7` | Q-0570 |
| `mod_addition_test.py` | `7786a0ce6e07bf5ee7f7de410a0a41536e22ff3e` | Q-0571 |

The [native tutorial](../../examples/notebooks/modular_addition.ipynb) retains
all five operations, all ten source signatures and the five diagram and
call-graph sections. Its 27 Python cells run in an independent kernel.
The catalog uses the existing modular, controlled, scaled and constant
addition compiler functions through `circuit.append(op)`. Source call
declarations preserve child widths, activation values, constant parameters
and the allocated symbolic constant for scaled addition.

## Bounded constant addition

The locked `ModAddK` specifies the permutation on computational states,
including identity for out-of-range encodings. Its source resource graph
declares five equal-width Add calls. Qualification of a synthesized gate
stream uses the value map because the upstream operation supplies no
physical decomposition.

The native bounded emitter computes `x < modulus` into a clean membership
bit, combines it with the optional control pattern, and uses fixed-width
controlled constant addition. On an active residue, the wrap bit records
`x >= modulus - k` for `k` reduced modulo the modulus. Conditional
subtraction of the modulus produces the final residue. The equivalent
predicate `result < k` erases the wrap bit. The output preserves
membership, so repeating the original comparison clears its flag.
Classical slots are reused after each measured operation has restored
its work. A full binary ring uses the existing constant adder directly.

The tests certify coherent restoration for every basis input and all
measurement histories in eight small configurations, including mixed
controls, identity cases, negative constants and the inverse value map.
The frozen 64-bit example also checks fitting residues, the wrap point,
the first padding word and the maximal word without allocating a dense
state. A two-bit native append fixture for each of the five operations
checks complex amplitudes and zero-work output on three feedback histories.
Existing concentrated suites retain the gate order and feedback validation
for the other four source arithmetic chains.

| Native example | Logical ports | Clean work | Reusable slots | Emitted CX | Emitted CCX | Emitted T/T† | Emitted measurements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ModAddK(4, 7, 1)` | 4 | 9 | 4 | 276 | 2 | 120 | 30 |
| `ModAddK(64, 500, 23)` | 64 | 129 | 64 | 4734 | 2 | 2040 | 510 |
| `CModAddK(4, 7, 1)` | 5 | 14 | 8 | 239 | 0 | 84 | 21 |
| `CtrlScaleModAdd(4, 7, 1)` | 9 | 15 | 8 | 980 | 0 | 352 | 88 |

## Remaining source-file boundaries

Q-0569 is accepted. Q-0570 retains the locked `ModAdd` target-at-modulus
cleanup counterexample, the `CModAddK` out-of-range constant disagreement,
and the absence of a source primitive decomposition for `ModAddK`.
The new bounded permutation is an exact realization of the latter's
documented classical map.

The locked suite passed 446 cases including the notebook and all 356 slow
cases, with 18 skips and four expected failures. The skips include symbolic
decomposition and strict register-type boundaries. Native numerical callbacks
match all input tuples of the three slow grids. The original controlled
constant tensor blocks are matched via the certified native output map.
Q-0571 is accepted. Q-0568 was subsequently accepted after the thirteen
package exports and their owners were audited. The later exact alias wire
evidence is recorded in the [package audit](MODULAR_PACKAGE_EXPORT_AUDIT.md).

## Stored graph interfaces and ownership

The next executable graph owners follow the unchanged four composite methods.
All words are MSB first. Arguments describe encoded integers, with a concrete
positive width and modulus in `[1, 2**width]`. Input quantum values retain the
source domain promise. No value inspection or source-domain repair is inserted.

| Owner | Ordered ports | Intermediate views and ownership |
| --- | --- | --- |
| `modular_addition_component.modular_addition_program` | `x, y`, Montgomery words with the declared modulus | Own junk/sign prefixes, widened Montgomery Add/AddK, source unsigned comparator views, release both prefixes |
| `controlled_modular_addition_component.controlled_modular_addition_program` | `ctrl, x, y`, uint or Montgomery, control polarity zero or one | Source Cast to unsigned, one sign prefix, CAdd, unconditional corrections, controlled comparator and release |
| `modular_constant_component.controlled_modular_constant_program` | `ctrl, x`, uint, positive control | Own IntState constant until matching IntEffect, preserve the original encoded constant without modular reduction |
| `scaled_modular_addition_component.scaled_modular_addition_program` | `ctrl, x, y`, uint, positive control | Iterate MSB bits, own one And output per bit until its measured inverse, call CModAddK with source-reduced scale constants |

Stored graph lifetimes use the shared scheduler and fresh measurement SSA slots.
The established explicit-wire emitters retain their separately accepted layouts.
`ModAddK` has no frozen executable decomposition and retains its independent
bounded physical permutation and five-Add analytical declaration.

## Complete definition disposition

Every definition in the unchanged source is assigned below. Concrete executable
closure and broader source-interface acceptance remain distinct.

| Source definitions | FlagQuantum owner or retained boundary |
| --- | --- |
| `ModAdd`, `ModAdd.signature`, `ModAdd.build_composite_bloq` | `modular_addition_component` and `ModularAdditionExample.signature`, declared Montgomery ports and original internal views |
| `ModAdd.on_classical_vals` | Existing `simulation.classical_modular_addition.modular_add_classical_values` |
| `ModAdd.build_call_graph` | Existing `ModularAdditionExample.call_graph` and independent modular resource models |
| `ModAdd.short_name` | Original notebook labels retained, no additional executable semantics |
| `ModAddK`, `ModAddK.signature`, `ModAddK._validate_mod` | Existing bounded modular owner, catalog signature and explicit executable validation |
| `ModAddK._classical_unctrled`, `ModAddK.on_classical_vals`, `ModAddK.__pow__` | Existing full-domain bounded permutation and signed integer multiples, original callback/control/power evidence reused |
| `ModAddK.build_call_graph` | Existing five-Add declaration, separate from synthesized primitive counts |
| `CModAddK`, `CModAddK.signature`, `CModAddK.build_composite_bloq` | `modular_constant_component`, shared IntState, controlled modular graph and IntEffect |
| `CModAddK.on_classical_vals` | Existing `simulation.classical_modular_constant_addition.controlled_modular_add_constant_classical_values` |
| `CModAddK.build_call_graph` | Existing catalog one-CModAdd declaration, separate constant Clifford model and primitives |
| `CModAddK.wire_symbol` | Original notebook's control and constant labels retained, generic diagram metadata reused |
| `CtrlScaleModAdd`, `CtrlScaleModAdd.signature`, `CtrlScaleModAdd.build_composite_bloq` | `scaled_modular_addition_component`, every MSB traversal and paired And lifetime |
| `CtrlScaleModAdd.build_call_graph` | Existing symbolic allocated-constant declaration, source model generalizes each concrete constant |
| `CtrlScaleModAdd.on_classical_vals` | Existing `simulation.classical_modular_scaled_addition.controlled_modular_scaled_add_classical_values` |
| `CtrlScaleModAdd.wire_symbol` | Original notebook's control and scale labels retained, generic diagram metadata reused |
| `CModAdd`, `CModAdd.signature`, `CModAdd.build_composite_bloq` | `controlled_modular_addition_component`, source unsigned casts and both activation values |
| `CModAdd.on_classical_vals` | Existing `simulation.classical_modular_addition.controlled_modular_add_classical_values` |
| `CModAdd.build_call_graph` | Existing catalog declarations and independent controlled modular resource model |
| `_mod_add`, `_mod_add_k`, `_mod_add_k_small`, `_mod_add_k_large`, `_cmod_add_k`, `_cmod_add_k_small`, `_ctrl_scale_mod_add`, `_ctrl_scale_mod_add_small`, `_cmodadd_symbolic`, `_cmodadd_example` | Existing ten catalog fixtures and accepted original notebook/test intake, concrete composable factories now exercise stored graphs |

All 40 class, method and factory definitions are disposed. The existing Q-0571
446-case original intake, including 356 slow cases, remains reusable evidence.
The stored-graph addition does not require rerunning those unchanged callbacks.

### Source failures and executable boundaries

The source's positive AddK loader requires an encodable constant. Consequently
modulus `2**width` cannot physically decompose any of the four composite chains.
The stored graph raises at that same dependency. The existing append catalog
retains its original full-ring rejection at this boundary. The separately
implemented ModAddK full-ring permutation remains available. The residue-range
catalog paths use stored graphs. A fitting CModAddK constant
outside `[0, modulus]` remains available for source diagnostics, while the catalog
continues its existing rejection. For `width=2, modulus=2, k=3, ctrl=1, x=1`,
source/native primitives produce 2 and the source classical callback produces 0.
No constant normalization repairs that source disagreement.

Target-at-modulus sign cleanup, strict type edges, symbolic physical admission,
foreign admission and source fixture count-mode limitations remain recorded gaps.
The concrete graph's loose-width compatibility retains the original untyped
Join and Montgomery-to-unsigned comparator connections. Five owned fixture modes
with no explicit qtyping edges do not establish strict source dtype acceptance.

## Stored graph acceptance and separate resources

The four concrete source graphs and their direct typed arithmetic, comparison,
basis-vector, cast and And dependencies are closed for the admitted executable
range. Root connections and full primitive streams, operands, individual
measurement slots and feedback match unchanged source through eight bits.
Both CModAdd control values and uint/Montgomery descriptors are covered.
Negative scale and zero-constant stages retain every original traversal.

All valid columns through three bits satisfy each certified And-erasure
invariant, proving identical amplitudes for every composed measurement history.
One- and two-bit coherent states include an entangled spectator. Every raw
history is enumerated when there are at most seven events. Three raw histories
are separately projected above that bound. Work is zero on the admitted domain.
The target-at-modulus dirty-sign witness and encoded out-of-domain constant
witness retain the unchanged source behavior. The existing catalog's normal
`circuit.append(op)` route binds stored graphs and passes with old physical
emitters disabled. Its original range refusals and independent ModAddK route
are preserved. Three actual concrete source factories pass all five owned
fixture modes. The qtyping mode has no explicit source edge proof.

For width eight, modulus 251, and constant/scale 173 where applicable,
primitive scans and independent source categories are as follows.

| Graph | Emitted CX | Emitted T/T† | Emitted X | Emitted measurements | Source And | Source Clifford | Native/source peak quantum | Fresh classical slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ModAdd | 376 | 124 | 27 | 31 | 31 | 248 | 34 / 34 | 31 |
| CModAdd, cv=1 | 479 | 164 | 45 | 41 | 41 | 320 | 35 / 35 | 41 |
| CModAdd, cv=0 | 479 | 164 | 81 | 41 | 41 | 320 | 35 / 35 | 41 |
| CModAddK | 479 | 164 | 55 | 41 | 41 | 320 | 35 / 35 | 41 |
| CtrlScaleModAdd | 3880 | 1344 | 422 | 336 | 336 | 2568 | 44 / 44 | 336 |

Stored Programs use one fresh SSA slot per measurement event. Earlier explicit
emitters reuse physical classical storage after cleanup. This slot distinction
does not change source gates, their operands or event-specific feedback.
Owned temporary quantum outputs are fully returned in every graph.

Local receipts use `ftqc-reference-env/g2-modular-native-*`. The final focused
lane contains 116 passing cases. The concentrated lane contains 8953 passing
cases with 22 established warnings. Its collection predates the final five
source-defect/catalog refusal additions, which pass in the final focused lane.
Six changed Python files pass strict typing, Black, Ruff, team scope,
architecture, IR and protected API checks. The default smoke/unit lane's
established docstring-coverage failure remains recorded. Its failing test and
nine reported modules are byte-identical to the prior published baseline.
The existing KMX primitive transport and original 446-case/356-slow source
intake are reused. G2 parent Q-0570 remains open for the specific boundaries above.


## ModAdd source-wire metadata closure

The original wire serializes x/y as width-only Montgomery words, while its
constructor signature includes the declared modulus. Both the canonical and
package-alias constructor now use the same existing `montgomery_port_views`
bridge already used by multiplication and inverse. The caller-facing signature
stays byte-for-byte faithful to the decoded wire. Explicit input Casts enter
the modulus-bearing shared arithmetic body; output Casts restore the external
views. No Core descriptor, serializer or source implementation is changed.

This bridge admits only the known omitted-modulus case. Eight negative signature
categories retain rejection: wrong explicit modulus, width, encoding,
quantum/classical kind, name, shape, side and count. Symbolic declarations keep
the source external signature and remain bodyless until execution can bind
concrete sizes. Explicit correct modulus signatures and direct numerical users
continue through their original path.

The actual primitive stream, output aliases and allocation/release instruction
offsets match the existing authoritative body. Logical call positions differ
because of the explicit Cast views and are not claimed identical. Original
source-walker operands and measurement feedback, valid-residue complex branches,
all-history work cleanup certificates, exact inverse evidence and separate
primitive/model resources remain authoritative. The source constructor and
package alias resolve to equal Programs and equal complete bound records.
This closes the wire-metadata gap; original target-at-modulus cleanup,
full-ring source loader failures, symbolic physical bodies and model-only
ModAddK are not relabeled as solved. Q-0570 therefore remains open.

## Complete-file closure after ordered symbolic-root delivery

The later [ordered modular-library audit](SYMBOLIC_MODULAR_GRAPH_AUDIT.md)
resolves the original finite symbolic-root gap with native constructor data and
all source connection endpoints. It also verifies that known-width Montgomery
symbolic interpretation metadata already follows the source physical path;
no extra arithmetic admission or altered source is needed. The original normal
append, complete stored constructors, exact adjoints, complex/all-history/work
and resource evidence above remains authoritative. The earlier generic
symbolic/foreign wording is superseded by this exact disposition, not by a claim
that unbound numerical leaves can execute.

All 40 source definitions now have native responsibility and direct dependency
closure. Under the approved faithful-migration rule, Q-0570 is eligible after
verified publication and final checks. Original target-at-modulus cleanup,
exterior constants, full-ring loader failures and strict/count/type outcomes
remain recorded source limitations. Atomic ModAddK still has no original gate
recipe; its native documented permutation and original analytical model remain
separate. Global SDK/transport certification is not inferred or waived.
## Fixed-width symbolic modulus constructor closure

ModAdd and CModAdd now retain every available fixed-width source call when only
the modulus is symbolic: seventeen and fourteen calls respectively. Shared
templates own widening, casts, addition, sign comparison and clean releases;
Formats supplies the actual unresolved minus-modulus and controlled plus-modulus
constructors through the existing constant catalogue. No numerical encoder is
invoked. Both control polarities and uint/Montgomery external words retain the
same source Add/CAdd and comparator specializations.

The frozen interchange codec omits Montgomery's modulus on signature records.
Its existing five-call Cast bridge is certified in full before comparing the
inner seventeen-call ModAdd owner. The bridge is not counted as an upstream
arithmetic decomposition. All available source call/operand layers are traversed;
atomic And views are certified against their exact shared bodies, and every
stopped source leaf must genuinely refuse decomposition. Symbolic constants
remain nonexecutable and supply no fabricated primitive, amplitude, cleanup or
default-resource evidence. Finite Program equality preserves the accepted
physical proofs. Symbolic word-width declarations remain a separate boundary.
## Exact symbolic modular addition adjoints

The fixed-width symbolic addition owner now reverses its original available
call graph using the accepted exact Add/CAdd and comparator inverses, explicit
unresolved constant adjoints and the same X leaf. Both controlled polarities
retain their original fourteen calls; ordinary ModAdd retains seventeen calls.
Widths 3/5 compare the actual source inverse parent and every new operand.
Complete unchanged arithmetic/comparison definitions are certified before their
existing primitive, branch, phase and work-zero evidence is reused. Each opaque
constant must genuinely refuse the unchanged source decomposition. This grants
no physical gate count, numerical inverse or cleanup proof at an unbound modulus.

## Literal additional controls and inverse protocols

The additional-control owner preserves ModAdd's seventeen calls and CModAdd's
fourteen calls, both original control polarities and exact inverse direction.
All available comparison, constant and arithmetic descendants remain explicit.
Full stored definition certificates reuse the accepted Always(Add/inverse)
owner before its phase and feedback evidence is reused; independent inverse
Split scheduling inside that unchanged owner is not claimed ordinal-identical.
Constructor source depths zero/one, SDK-free reconstruction and negative outer
predicate wrappers retain the original ports and source modulus-codec bridges.

The source control protocol differs by direction. Forward Add selects CAdd,
and forward CAdd selects the three-call singly-controlled reduction. Generic
control on Add-adjoint or CAdd-adjoint instead propagates control into their
inverse bodies. Those inverse parents are available, but their controlled And
leaves refuse non-THRU target ports. They remain declarations; a numerically
valid specialized inverse is not substituted for the original generic recipe.
The wider controlled comparator's original directional refusals remain explicit.
No complete amplitude, measured gate inventory or work-zero claim covers these
unbound modular parents. Symbolic added controls and inherited full-ring or
symbolic-width construction boundaries remain open.

The independently executable three-call arithmetic specialization has complete
complex operators, entangled spectators, every small raw measurement history,
zero-work and append execution evidence. For a=2,b=3 and controls=(1,1), both
directions emit H15,T10,CX38,Tdg10,S5,measure5,CZ5,reset5. The separate original
model is And5,Toffoli0,Clifford14,measure5,T-equivalent20,peak qubits12.
For the one-bit data case with controls=(1,0), actual primitives include four X,
six H, thirteen CX, four T and four Tdg, plus two each of S/measure/CZ/reset.
The original model reports Clifford=-1 there; this source model anomaly is
retained separately and is not a physical negative gate count.
