# Trigonometric source family whole-file audit

Q-0223–Q-0228 cover the entire arithmetic trigonometric package at
Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`. The
six original Git blobs match the frozen inventory.

## Ownership, word contracts and source disposition

| Original definition | Native ownership and tested scope |
|---|---|
| Package initializer | Its only statements re-export ArcSin and ArcTan. ArcSin routes to Simulation for scalar values and Compiler fixed point for Appendix D lowering and resource declarations. ArcTan routes to Compiler lookup for exact bounded words, fixed point for the independent arithmetic approximation, and the shared trigonometric resource model for the original estimate. These routes use existing package boundaries. Foreign class exports remain an interface difference of the implementation files. |
| ArcSin fields, validation and signature | The complete ten-bit factory signature retains THRU x and result, both unsigned fixed point with seven fractional bits. The native neutral dtype validates fixed-point shape. The example declares the original iteration and degree defaults through its model. Generic foreign constructor admission remains open. |
| ArcSin.on_classical_vals | Simulation preserves the scalar full-width fractional interpretation, binary64 domain check, truncation and occupied-result XOR. The declared fractional width is absent from this source calculation. Word 15 at width four produces 19 before output encoding. The native scalar result remains unmasked. Verified all valid words at widths 1/2/3/4/5/8/10 under several declared splits. |
| ArcSin.build_call_graph, factory and documentation | Compiler retains the independent Appendix D Toffoli equation, checked over 36 concrete parameter combinations and unrestricted symbolic expressions. The complete source instance declares 13425 Toffoli. No source primitive decomposition exists for it. |
| ArcTan signature, registers and with_registers | The example declaration retains the opaque select and target and QBit sign, including symbolic widths. The original Cirq register tuple and symbolic/rebinding failure boundaries are recorded in the tests. Native execution binds wires explicitly. |
| ArcTan.apply and self-adjoint | The existing bounded QROM preserves big-endian unsigned select, sign XOR and fractional target XOR. All six original width pairs now compare the complete supported input domain at zero and occupied outputs to the locked callback. Certified primitive branches verify the outputs, zero work and two applications restoring the input. Existing three smaller source-derived QROM cases retain direct gate, operand, feedback and complex-state evidence. |
| ArcTan.build_call_graph | Retain one PlusEqualProduct with operand widths n/n/2n and fresh source multiplicity represented by the original single-query symbol. The symbolic legacy estimate is 8 c n². The original has no corresponding product gate decomposition. |

The independent Appendix D ArcSin circuit accepts a big-endian
nonnegative input in [0, 1] with one integer bit and produces a separate
one-integer-bit output. It uses a quantized factor polynomial and
explicit half-pi, XORs its result into a bound target, and restores clean
scratch. The signed wrapper, fixed-point reduction, inverse square root,
Horner and measurement feedback evidence are reused. Coefficients are
independently reproduced with a Chebyshev projection. The original
minimax coefficient table is still missing. The original scalar callback
and its overflow witness establish a different word contract from this
executable approximation.

The independent ArcTan arithmetic path retains its existing reciprocal
reduction and Horner evidence. Four of 256 input words in the previous
eight-bit comparison differ from the original by +1 ulp. The bounded
QROM path supplies exact locked-word parity within its admitted range.

## Original test and resource accounts

Four original test functions execute, including all original slow
ArcTan cases. The frozen run reports 10 passed and 3 skipped. The skips
are ArcSin decomposition, count comparison without a decomposition,
and dtype checks requiring decomposition. No reference file was edited.

| Path | Recorded account |
|---|---|
| Original ten-bit ArcSin factory | 13425 Toffoli in the declared model |
| Independent Appendix D teaching circuit | 6856 T/TDG, 16918 CX, 88 CCX, 1714 measurements, 196 work wires and 13 reusable slots |
| Original symbolic ArcTan | 8 c n² T in the product estimate |
| Native three-bit-to-three-bit ArcTan QROM | 16 T/TDG, 43 CX, 4 measurements, 2 work wires and 2 reusable slots |

The Appendix D teaching circuit is emitted through `circuit.append(op)`
and reported as IR. Its branch and coherent sparse execution evidence
comes from the dedicated native tests. The notebook independently
simulates the complete complex ArcTan input/occupied-target state and
its restoration through append. Emitted primitive counts, source
declared costs and polynomial factor error are distinct accounts.

## Teaching asset and closure

The original ten-cell notebook covers ArcSin only. The native
fourteen-cell notebook preserves all original topics, signature and
graph, and adds the scalar/Appendix D distinction and an ArcTan
resource/lookup example. Both assets pass independent kernels. The
native kernel blocks foreign reference imports.

Q-0223 closes through the complete two-export responsibility map.
Q-0228 closes as a complete teaching asset. Q-0224–Q-0227 retain the
original decomposition, generic interface, approximation and original
test-mode boundaries above.

## Complete generic declarations and foreign consumers — 2026-11-20

All 13 definitions across arcsin.py and arctan.py now have explicit owned
interfaces or preserved original-source failures. The existing original scalar,
lookup, approximation, notebook and original-test evidence remains authoritative.

| Source responsibility | Owned disposition |
|---|---|
| ArcSin fields/defaults/post-init/signature | `fixed_point/arcsine_declaration.py` retains width/fraction/iteration/degree metadata, defaults, exact source validation including equality, and unsigned fixed-point x/result. Concrete and symbolic signatures match original serialized descriptors. |
| ArcSin classical callback and direct cost | Existing Simulation callback and independent source formula are reused. The actual width-ten/fraction-seven factory declares 13,425 Toffoli. |
| ArcSin factory and documentation | Actual concrete/symbolic libraries transport exact source bytes at depths 0/1/2, reconstruct equally upstream and resolve to native typed declarations. Existing complete notebook and factory evidence is reused. |
| ArcTan signature and registers | `lookup/arctangent_declaration.py` owns select/sign/target and original binary digit dimensions. Symbolic digit construction retains TypeError. Original with_registers remains unimplemented. |
| ArcTan apply and adjoint | Existing bounded QROM retains the actual unsigned encoded domain, occupied XOR outputs and self-inverse mapping. Native declaration adjoint returns itself. Original tensor-only SDK import is qualified separately below. |
| ArcTan direct model | The generic declaration uses the shared SymbolAllocator, producing fresh c multiplicities per query. Catalogue delegates to the declaration. Sequential original/native allocators agree exactly. |

The complete direct dependency closure also repairs two shared format defects.
The original ArcTan library contains `_c0`, previously refused by the native
parser. Single-underscore atoms now construct symbolic nodes directly, with
existing dunder/call/attribute/subscript refusals retained. The symbolic ArcSin
model exposed intermediate product distribution that changed printed wire
expressions. One product construction now preserves the original factor grouping.
Both complete libraries retain exact source bytes in all three depth settings.
See `PROTO_ARGUMENT_FILE_AUDIT.md` for the format ownership and regression scope.

Both sources lack gate decomposition. ArcSin also lacks a tensor-unitary SDK
fallback, and the existing import reports unsupported_operation. Its width-four
input word 15 produces 19 for all declared fractional widths 2/3/4. Those three
original failure witnesses remain unchanged. ArcTan's register-aware gate helper
returns no decomposition. The existing operation-tree import with explicit typed
placements then consumes the original ArithmeticGate tensor as one matrix leaf.
No new SDK import policy or Core contract is introduced.

Actual ArcTan SDK tensor consumers at 1/2, 2/3 and 3/3 source/target widths agree
in complete complex amplitude with a coherent spectator. Native QROM consumers
agree with that same original matrix in every raw measurement history, including
zero work after each branch. The QROM's primitive stream and resource model
reuse accepted original-QROM evidence. The tensor leaf is one `any` operation.
At 3/3 the native QROM emits 43 CX, 16 T/TDG and four measurements, with two clean
work wires. The original ArcTan product estimate is independently 72*c T. Matrix
size, primitive emission and symbolic arithmetic estimates stay separate.

Acceptance has 551 concentrated passes. The initial twelve optional QREF skips
were rerun with the existing locked qref-bartiq directory and all twelve pass.
All 17 integration cases pass. The 25 declaration, ten foreign and 46 shared
argument cases also pass together on Cirq 1.6.1, with twelve original NumPy shape
warnings. Strict typing covers seven changed Python files. Black, Ruff, scope,
architecture, IR and public API baseline checks pass. Frozen source hashes and
unchanged numerical/SDK owners are verified in `g2-trigonometric-foreign-evidence.json`.

Q-0225 and Q-0227 close with original 10-pass/3-skip source evidence retained.
Q-0224 remains open for the unrecovered original minimax coefficient obligation.
At that publication, Q-0226 remained open for its generic callback and the
independent arithmetic path's four of 256 +1-ulp words. Their missing original
gate bodies remain explicit source limitations.
This batch completes generic interfaces and foreign consumers without claiming
those independent scientific gaps are resolved.

## Stored exact lookup and explicit constructor replacement — 2026-11-25

The complete ArcTan source was reread at Git blob
`746411d9f44fa95c5209f5575419a99749937e99`. Its fields, signature, digit
dimensions, unsupported rebinding, mathematical XOR, self-adjoint declaration
and fresh-symbol product model retain the owners above. There is still no
upstream primitive decomposition to copy or count.

`lookup/arctan_lookup_component.py` now stores the existing exact bounded lookup
as a native Program. It reuses the table rounding and admission owner, the typed
QROM and its measured predicate cleanup; it adds no numerical execution kernel.
Selection, sign and target are THRU, MSB-first words. Occupied outputs are XORed,
selection is preserved, and QROM privately owns and erases every work bit.
The existing profile remains 1–12 input bits and 1–32 target bits. The earlier
4096-row, 12/32-bit source quantization certificate is reused.

The default `trigonometric_factories()` still returns a bodyless source ArcTan
declaration. Consumers may explicitly replace its ArcTan entry with
`arctan_lookup_factory`; the existing resolver and caller remain unchanged.
Actual source libraries at depths 0/1/2 bind to the same complete Program as
direct construction. Invalid signatures, extra arguments, symbolic execution
and unsupported widths fail instead of falling back. This explicit native
extension is not labeled as an upstream physical body.

Four profiles compare the complete emitted stream against the unmodified
source QROM, including per-wire order, operands and measurement feedback.
Source measurement events map to native SSA results: a reused flat classical
slot and distinct stored SSA slots are different names for successive results,
not additional logical measurements. Full complex vectors with occupied sign
and target plus a coherent spectator match the original ArcTan matrix in every
small raw history. Two composed lookups restore the input in every applicable
history, with all work zero. A cold process blocking Qualtran and Cirq builds,
binds and appends the native Program through the existing operation protocol.

| Input / target bits | Emitted CX | Emitted T + Tdg | Emitted measurements | Clean work | QROM model T / measurements | ArcTan model T |
|---|---:|---:|---:|---:|---|---|
| 1 / 1 | 2 | 0 | 0 | 0 | 0 / 0 | 8 c |
| 2 / 3 | 22 | 8 | 2 | 1 | 8 / 2 | 72 c |
| 3 / 8 | 79 | 24 | 6 | 2 | 24 / 6 | 512 c |
| 4 / 6 | 149 | 48 | 12 | 3 | 48 / 12 | 288 c |

The actual inventory and independent models are recorded in
`g2-arctan-program-resources-20261125.json`. ArcTan primitive counts remain
unavailable, not zero. This closes the stored lookup/constructor execution
path. At that publication Q-0226 remained open with the separately implemented arithmetic
approximation's four of 256 +1-ulp words; it does not mean the exact table has
four wrong rows. No parent completion or constructor-count increase is claimed.

Final acceptance: 175 concentrated lookup/QROM/trigonometric cases, 64 applicable
Cirq 1.6 cases, 10 unchanged original passes with three original atomic-mode
skips, and 145 local integration cases pass. Strict typing covers all four
changed Python files. Format, lint, scope, language, architecture, IR and public
API checks pass. The dependency scan retains the same 720 existing findings,
ignoring line-number movement; it is not globally green. Receipts use the
`g2-arctan-program-*-20261125` prefix in the reference evidence directory.

## Empty lookup dependency closure

The explicit native lookup now accepts input widths 0 through 12 and magnitude
widths 0 through 32. The original ArcTan signature omits empty select/target
ports but always retains sign. At input width zero, its complete arithmetic
operator is identity, including occupied outputs; the stored Program therefore
returns its live inputs without work or calls. For a nonempty input and empty
magnitude, both native emitters reuse a single sign dataset in the existing
QROM. MSB order, shared predicate ownership and measured cleanup remain there.
No artificial zero-width target is passed to QROM and no source body is invented.

Actual source-library depths 0/1/2 resolve through the explicit replacement
factory with the same signatures and emitted gates as direct construction.
Source QROM gate order, operands and feedback are checked for sign-only widths
1, 2 and 3. Full complex source matrices, coherent external spectators, every
small measurement history and applicable double-lookup cleanup are checked for
empty-input and sign-only profiles. Existing positive-profile Programs and flat
IR are exactly equal to published f9a824789 at 1/1, 2/3, 3/8 and 4/6.

| Input / target | Emitted CX | Emitted T + Tdg | Measurements | Reserved clean work | ArcTan model T |
|---|---:|---:|---:|---:|---|
| 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| 0 / 3 | 0 | 0 | 0 | 0 | 72 c |
| 1 / 0 | 1 | 0 | 0 | 0 | 0 |
| 2 / 0 | 9 | 4 | 1 | 1 | 0 |
| 3 / 0 | 17 | 8 | 2 | 2 | 0 |

These are measured native counts, separate from the original symbolic model.
The model's zero at empty target is not a claim of zero physical work. Native
lookup and the independent approximate arithmetic path retain separate scopes;
Q-0226's closure is reviewed below; the latter's quantization differences remain. Receipts
use `g2-arctan-empty-*-20261125`.

## Complete scalar callback and source-definition review

Planning section 2 distinguishes source formula-only components from separately
specified executable decompositions. G2 batch rule 5 also preserves existing
source-defect acceptance requirements. This review records both scopes; it does
not declare the original minimax table recovered or the independent arithmetic
path bit-exact. The following table accounts for all thirteen source definitions.

| Frozen definition | Native owner or explicit source disposition |
|---|---|
| ArcSin | `ArcsineDeclaration` fields, immutable metadata and defaults |
| ArcSin.__attrs_post_init__ | Declaration validates the original fraction/width relation; symbolic validation remains deferred |
| ArcSin.signature | Explicit named x/result THRU ports, including empty words, in the same declaration |
| ArcSin.on_classical_vals | `simulation.classical_arcsine.source_arcsine_classical_values`, including unmasked overflow and binary64 assertion |
| ArcSin.build_call_graph | Declaration delegates the unchanged direct Toffoli polynomial to `trigonometric_resource_model` |
| _arcsin | Existing actual factory/library depths 0/1/2, native example, graph and complete notebook disposition |
| ArcTan | `ArctangentDeclaration` fields and immutable metadata |
| ArcTan.signature | Declaration preserves select/sign/target and the original omitted empty ports |
| ArcTan.registers | Declaration's binary digit dimensions, including symbolic refusal and empty tuples |
| ArcTan.with_registers | Original hook raises NotImplementedError; no native dimension-changing gate operation is invented |
| ArcTan.apply | New Simulation normalized-angle owner plus Ecosystem scalar XOR callback and the existing real fixed-point codec |
| ArcTan.build_call_graph | Declaration preserves fresh symbolic multiplicity of PlusEqualProduct, independent of physical lookup counts |
| ArcTan.adjoint | Declaration returns itself; existing occupied-target quantum lookup proves its XOR inverse separately |

The actual missing original feature was generic ArcTan.apply. Its native
counterpart now composes `simulation/classical_arctangent.py` for binary64
normalization with `ecosystem/classical_arctangent.py` for the scalar XOR and
existing optional fxpmath conversion. Ecosystem retains external-object and
optional-dependency ownership. Simulation imports no fxpmath or Ecosystem;
Compiler receives no numerical execution dependency. The existing reviewed
fixed-point conversion contract and entry points are reused. No Core export,
IR vocabulary, public schema or quantum lookup bound changes.

The callback retains the input integer and does not mask occupied sign/target
values. It has no 12-bit input/32-bit target lookup restriction. The 33-bit
target profile executes exactly like the original. In ten target profiles
0/1/2/8/32/33/53/64/80/128, the same fourteen scalar selections produce the
same results or exception classes. NumPy refuses a sufficiently large Python
integer; the locked fxpmath version also overflows on some 64/80-bit conversions
and all tested 128-bit conversions. These are explicit source failures, not
silently skipped inputs. A large finite selection rounds the angle magnitude
to one in binary64; original wrapping yields zero rather than clipping to the
maximum encoded word. Both native normalization and conversion preserve it.

Scalar successes cover occupied and exterior raw outputs, both signs and double
application. Reference-blocked execution demonstrates that neither Qualtran nor
Cirq is needed. Removing the optional codec still allows Simulation math and
causes the explicit Ecosystem conversion to fail, rather than substituting a
different quantizer. Source-comparable small quantum operators retain ordinary
append, complete complex spectator amplitudes in every raw history and zero
work. Classical callback evaluation is not assigned a zero-gate quantum count.
Independent source resource formulas and measured QROM primitive inventory keep
their existing separate owners.

The additional ArcSin intake compares source success/failure and unmasked values
at widths 0 through 1024 without finding a new mismatch. Existing whole-small
domain and symbolic callback tests are reused. It does not claim exhaustive
enumeration at large widths. Source/test blobs, actual factories, notebook and
SDK consumers retain their accepted evidence.

Parent conclusions remain explicit: Q-0224 retains its open original-coefficient
and separately registered chemistry/decomposition obligations. Q-0226 closes
only through the exact callback and quantum operator coverage below, not by
declaring the independent approximation exact. Its recorded four differing words
remain unchanged. No missing source primitive stream is invented.
Final receipts use `g2-trigonometric-scalar-*-20261126`; the November 25 intake retains the original success/failure samples.

## ArcTan complete source-file closure

Q-0226 now has native ownership and executable evidence for every original
definition, its generic scalar callback, typed declarations, symbolic estimate,
original SDK operator and unsupported dimension-changing hook. The exact
callback composes with the existing generic typed QROM for quantum execution;
it is independent of the narrower convenience lookup profile. This uses the
existing-equivalent-capability disposition in planning section 2, rather than
substituting a new approximation or requiring a nonexistent original primitive
body. Source exceptions, original test assets and declaration-only primitive
lowering retain their explicit dispositions.

The 13-input-bit/one-target-bit operator covers all 32768 occupied logical basis
inputs with measured-And cleanup certificates, compares every primitive operand
and feedback event with the unchanged original QROM, and preserves a coherent
sample with a spectator under sparse execution. This is not a claim of dense
simulation over all physical work states or raw-history enumeration.
The one-input-bit/33-target-bit operator is X/CX-only with no phase or work. Its
zero and basis-vector images determine the complete affine occupied-target
operator, and coherent sparse execution preserves the corresponding amplitudes.
Both datasets equal literal original ArcTan.apply values. Consumers retain
ordinary circuit.append(op); no new Core instruction or gate API is needed.

The independent polynomial-size implementation remains explicitly approximate:
four of 256 words differ by one ulp at the previously certified profile. Its
existing scoped acceptance is not changed, and it is not used as the exact
source-file replacement. Q-0226.02 records verification that the original only
supplies its count formula and ArithmeticGate callback, not a primitive recipe.
Q-0224 and all its open coefficient/chemistry work remain open. Closing this
one source file does not complete G2 or waive any other parent requirement.


## Published-batch verification and resource separation

The concentrated regression passes 290 cases; the Cirq 1.6 lane passes 170,
and local integration passes 145. Strict typing of the four changed Python
files, formatting, lint, both owner scopes, language, links, architecture,
Program IR and public API checks pass. The frozen-source audit covers all
thirteen definitions above and verifies the real Kickmix 0.1 Simulator.

| Exact generic QROM profile | Measured primitive inventory | Reserved clean work | Independent source models |
|---|---|---|---|
| select 13, target 1 | 26 X, 36 H, 24 T, 24 Tdg, 110 CX, 12 S, 12 measure, 12 CZ, 12 reset | 12 | QROM: 12 And, 52 Clifford, 12 measurements; ArcTan: `8*_c0` T |
| select 1, target 33 | 2 X, 2 CX | 0 | QROM: 4 Clifford; ArcTan: `8712*_c0` T |

The QROM count graph and its measured decomposition use different categories.
The ArcTan estimate retains the original fresh unknown multiplicity `_c0`;
it is not a prediction of this table-based implementation. The scalar callback
has no measured quantum inventory. Receipts include the complete primitive
comparison and `g2-trigonometric-scalar-resources-20261126.json`.
