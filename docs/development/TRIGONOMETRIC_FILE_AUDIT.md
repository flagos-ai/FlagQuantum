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
Q-0226 remains open for the independent arithmetic path's four of 256 +1-ulp
words. Their missing original gate bodies remain explicit source limitations.
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
path. Q-0226 remains open for the separately implemented arithmetic
approximation's four of 256 +1-ulp words; it does not mean the exact table has
four wrong rows. No parent completion or constructor-count increase is claimed.

Final acceptance: 175 concentrated lookup/QROM/trigonometric cases, 64 applicable
Cirq 1.6 cases, 10 unchanged original passes with three original atomic-mode
skips, and 145 local integration cases pass. Strict typing covers all four
changed Python files. Format, lint, scope, language, architecture, IR and public
API checks pass. The dependency scan retains the same 720 existing findings,
ignoring line-number movement; it is not globally green. Receipts use the
`g2-arctan-program-*-20261125` prefix in the reference evidence directory.
