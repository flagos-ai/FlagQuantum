# SELECT and LCU block encoding source files

The scoped typed constructor path and its independent control-adder ownership
are documented in [the constructor audit](RESOLVER_BLOCK_ENCODING_AUDIT.md).
The Algorithms execution receipts below remain valid evidence for their own
path; they do not close the original constructor scalar-body or serializer
boundaries recorded in that audit.

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen files under
`qualtran/bloqs/block_encoding/` are Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0282 | `lcu_block_encoding.ipynb` | `4f457c91bedc3baaf94f7515242a67fffc4bf4b7` | Closed by the native tutorial. |
| Q-0283 | `lcu_block_encoding.py` | `679ac7964baf62288713078a5ec4dbce674cbece` | Native through/RIGHT oracle replacement and lifetimes verified. The original LEFT-junk extraction failure remains open. |
| Q-0284 | `lcu_block_encoding_test.py` | `0632799611bbd5ec2b7ca724533d4bbd1bbb7169` | Closed by the original-fixture mapping and existing controlled branch evidence. |

## Binding and work ownership

SELECT-only exposes PREPARE as the signal state and emits one SELECT. The
LCU wrapper emits PREPARE, directional SELECT, then PREPARE adjoint, with an
identity zero-signal preparation. Specialized control changes SELECT alone.
The Hubbard catalogue binds the existing PREPARE and SELECT direction
callbacks. Packed registers use the existing partition helper in source
order. This regrouping emits zero gates. The named and packed forms give
identical physical instructions, operands and classical conditions at
the same binding.

For the original 3 × 3 parameters, the selection, preparation junk, and
system widths are 12, 2, and 18. Both representations have normalization 72
and source error metadata zero. PREPARE owns the two logical direction
bits. SELECT does not receive those bits. Each stage owns its temporary
work and result slots during the call. The sequential stages can reuse the
same physical scratch allocation. Explicit preparation-local outputs instead
stay live through SELECT and cannot share its scratch until unpreparation.

## Typed declarations and drawing

The SELECT declaration preserves the selection and target descriptors.
PREPARE junk remains part of the independent signal preparation. The LCU
declaration starts with the optional scalar QBit control, retains the
PREPARE selection, includes only THRU junk, and then exposes the SELECT
target. Descriptors retain their shape and dtype. The direction-filter
test covers both LEFT and RIGHT junk and a retained register array.
This is a signature check. Physical bindings still require caller-owned
scratch and compatible branch cleanup.

The named Hubbard example keeps bounded integer selection words and an
opaque target. The packed example keeps opaque selection, junk and system
words. The notebook uses these typed declarations for its signature
figures. Drawer callbacks preserve the empty connector, `B[H]` boxes and
the source's literal `control` glyph rule. The actual source control port
is named `ctrl` and renders as `B[H]`. A port named `control` renders as the
filled or empty circle for its activation value.

## Original scenarios

| Original tests | Native counterpart |
| --- | --- |
| `test_lcu_block_encoding`, `test_black_box_lcu_block_encoding` | Original 3 × 3 metadata, ordered parameters and width, one PREPARE/SELECT/PREPARE-adjoint each, independent QEC/peak model, emitted CircuitIR codec round trip, exact named and packed streams for both directions and both control polarities. |
| `test_select_block_encoding`, `test_black_box_select_block_encoding` | Same original Hubbard metadata, one modeled and executed SELECT, exact named and packed emission in both directions and emitted CircuitIR codec round trip. The preparation remains available as the independent signal state. |
| `test_ctrl_lcu_be_cost` | Native positive and negative control models match the unchanged source. The positive case declares 28 Fredkins, 77 And operations, 294 Cliffords, 16 rotations and 77 measurements. |
| `test_notebook` | The frozen 40-cell notebook and the 42-cell native tutorial independently execute. The native tutorial contains eight signature and call graph figures and checks the compiled positive-control operation. |

The original non-notebook run has 17 passes and four mode skips for
decomposition-derived counts. The native catalogue matches the source fixture
metadata and modeled costs at 3 × 3. Existing unit tests compare original
PREPARE and SELECT decomposition details and all small SELECT/LCU composite
histories. The separate packed Hubbard 2 × 2 trial compares a complex
projected state with the independent Hamiltonian action and checks every
temporary bit in four representative measurement histories. Full output
parity for a coherent input to the original 3 × 3 block is recorded below. The codec
verification applies to emitted native IR.

## Original 3 × 3 full complex output

The native packed LCU is appended in both directions, without an outer
control and with each control polarity. The controlled case puts the
outer control in a coherent superposition. Each case compares every
occupied output amplitude on system, selection and logical junk with
the unchanged named source export. Both executions verify balanced
measurements and zero private work. The six native work bits are reused
across the three stages. Four histories are used in each case, all-zero,
all-one and both alternating patterns, with independent lengths for the
source and native streams.

The locked Cirq export retains a two-bit `ModAddK` leaf for the
increment modulo three. The source interpreter applies the source
classical permutation linearly and uses the source negative power for
the wrapped adjoint. At modulus three the adjoint must decrement. The
same shortcut as the one-bit increment would remove the hopping terms
from the projected reference result. The remaining source operations
use their exported Cirq matrices and measurement feedback.

The independent source export and native primitive stream differ in
their lowering of modular increment, swaps and multi-control arithmetic.
The accepted named and packed native streams remain identical. The
full complex comparison covers the complete resulting support for the
specified coherent input and four histories in each direction and control
configuration. The through-port replacement below adds a full small-input
operator comparison and long-history certificates. It does not claim enumeration
of all large input vectors.

## Emission and source model

| Original size | Emitted T/T-dagger, measurements, reserved wires | Source Fredkins, And operations, measurements, peak logical qubits |
| --- | --- | --- |
| 3 × 3, no control | 540, 86, 38 | 28, 74, 74, 37 |
| 3 × 3, positive control | 552, 89, 39 | 28, 77, 77, 39 |

The physical Hubbard stage expands its bounded arithmetic and Fredkin
decompositions. The locked source model uses abstract charges for those
children. Both columns retain their own counting and scheduling conventions.

## External instruction dependencies

The optional feedback importer maps exact unshifted half-X powers to the
existing native `sx/sxdg` and positive-controlled `csx/csxdg` opcodes. Their
absolute scalar phase is retained. The shifted rotation convention remains
`rx`; it is not silently replaced by a square-root-X instruction. Source
instrument comparisons cover both inverse powers and measured feedback.
No public static importer contract or protected gate vocabulary is changed.

The locked `ModAddK` supplies a total classical permutation and a declaration
of five additions, without a physical decomposition. Its optional conversion
uses the existing native bounded increment owner for literal steps `+1/-1`
and the bounded modular addition owner for other constants. It does not
contract an arbitrary source object into a matrix or manufacture a reference
gate stream. The original adjoint wrapper resolves through the source's own
negative integer power before native binding, not by reversing measurement
instructions. All binary padding states remain fixed.

The bridge owns its clean work from allocation through complete expansion and
releases it in `finally`. Existing register export handles quantum placement
and feedback. Bridge-private key tags are removed before import; keys are
assigned deterministically after reserving every foreign measurement name.
They cannot shadow the original records. Repeated capture produces identical
native JSON, and a colliding source-key fixture retains distinct conditions.
Native SDK helper objects do not enter the returned IR.

All binary inputs and a complete coherent input are compared against the
unchanged `ModAddK.on_classical_vals` contract for unit increments, decrements,
a nonunit constant, full binary modulus, modulus one and both control
polarities. The native history certificate proves every measured branch and
clean private work; numerical reference checks include the full binary input
basis. Source `mod_addition.py` remains blob
`0a167e84fe00d5df50488c4b28925ea6114728f7`.

All four original named/packed SELECT and LCU objects now capture actual
forward and adjoint instructions. Both LCU activation values also pass capture,
native IR JSON roundtrip, deterministic recapture and every-history cleanup
proof. The source typed register order is preserved, with big-endian bit order
inside each word. SELECT has 30 logical wires; PREPARE's two logical junk bits
are absent from its execution signature. The LCU has 32 logical wires, or 33
when controlled. Returned-wire binding swaps are zero.

| Captured source mode, named or packed | Reserved wires | Instructions | Measurements | T/T-dagger |
| --- | ---: | ---: | ---: | ---: |
| SELECT | 35 | 1134 | 40 | 230 |
| LCU | 40 | 2526 | 86 | 540 |
| LCU, control on zero | 41 | 2578 | 89 | 552 |
| LCU, control on one | 41 | 2576 | 89 | 552 |

Both directions retain these figures. Frame lifetime protection prevents
descendants from allocating any sibling Qid still present in the exported
circuit; this reserves two additional private names without adding gates.
The source abstract model still reports
74 or 77 measurements and its own logical peak as shown above. The added
physical work and erasures belong to native realization of the declared
arithmetic. An earlier generic realization used more work for unit steps;
the accepted path reuses the same unit-step owner as native Hubbard PREPARE.

These dependencies pass 166 concentrated feedback, modular, source-root,
block-snapshot and sparse-oracle regressions. Strict checks cover the three
production files and shared certificate; format and architecture checks are
recorded with the publication. Instruction capture and history proofs alone do
not replace native providers. The replacement below closes the concrete
through-port path. Preparation-local sided junk ownership remains open. The
existing signal PREPARE must stay nonidentity for SELECT. Its SDK object must
not become native metadata.

## Owned PREPARE and SELECT provider replacement

`CircuitRegisterEmitter` implements the existing named callback used by both
PREPARE and SELECT. It owns a layout, two native `CircuitIR` snapshots and
explicit returned-wire swap counts. No SDK oracle or bound SDK callback survives
conversion. The optional `import_register_emitter` captures the source forward
and source adjoint separately. It never reverses a measured instruction stream.
The shared `_storage` and `_emit` owners validate and bind workspace, classical
slots and physical output ordering; no second allocation or resource engine is
introduced. This is a local single-device execution boundary.

The existing `BlackBoxPrepare`, `SelectBlockEncoding` and `LCUBlockEncoding`
consumers are unchanged. Tests replace their callbacks with the captured owned
emitters and still append `native.make_on(...)` to an ordinary `Circuit`.
Logical ports follow the source register order; array elements flatten row-major
and bits within words remain big-endian. The existing typed declaration/drawing
path retains dtype and side metadata independently of this physical flat layout.
Selection, through junk and target remain logical outputs. Scratch starts after
those ports. Sequential preparation, selection and unpreparation reuse the same
clean work and result storage. Specialized source control names are read from
the captured signature: the two activation polarities need not spell their
control ports identically.

All original named and packed 3 × 3 LCU cases, in both directions and all three
control modes, pass replacement and every-history cleanup certificates. The
physical instruction, measurement, T-family and reserved-wire figures remain
exactly those in the capture table above. SELECT-only replacement has 30 logical
wires and five private wires; its nonidentity signal PREPARE retains its own two
logical junk bits and is not executed by SELECT. The captured LCU now reserves eight private wires after complete exported-frame
lifetime protection; its gates and measurements remain unchanged. The existing
hand-written Hubbard implementation still uses its original work layout.

For each small source case, every basis column, arbitrary logical through-junk
inputs and a complete complex input vector are compared with the unchanged
source tensor for every measurement history, in both directions and control
polarities. The original 3 × 3 replacements compare all occupied complex output
amplitudes, including coherent outer controls, system and selection input, with
the independently exported source. Two native histories are executed per case;
the certificate proves branch independence and zero private work for every
history. These large input vectors are sparse coherent probes, not an exhaustive
enumeration of the 32/33-bit logical input space. The comparison takes the union
of both numerical supports so differing interpreter roundoff pruning cannot
hide an output amplitude. Only individual local native unitaries are converted
to matrices for the test interpreter; whole blocks are never contracted into a
matrix as an import fallback.

The default emitter admits matching through-port circuits. Its explicit
`local_ports` request now also binds RIGHT outputs to clean caller storage and
captures their consuming adjoint separately. These ports are not reclassified
as logical THRU junk. The lifetime closure and remaining source diagnostic are
described below.
Independent source models remain caller-supplied declarations; a snapshot does
not manufacture an estimated cost.

Validation: 183 concentrated tests and 60 local CPU/representation integration
tests pass, without skips. The independent native LCU notebook kernel also
passes. Strict typing covers both production modules; Black, Ruff, both ownership
scopes, architecture, multi-level IR and public API checks pass.

Runnable replacement evidence:

```bash
pytest -q tests/unit/test_lcu_oracle_binding.py
```

## Directed local-output lifetime closure

The Algorithms-owned `BlackBoxPrepare.local_junk_layout` describes persistent
storage at the front of `prepare_workspace`. Its emitter receives those named
ports separately from the remaining scratch. PREPARE starts with clean local
storage, leaves its declared outputs live, and its original adjoint consumes
them after SELECT. `LCUBlockEncoding` rejects SELECT scratch that overlaps this
prefix. Its logical `junk_layout` still includes only THRU junk, while source
`resource_bitsize` includes both THRU and preparation-local storage. SELECT can
also use a named preparation-local output when the source declares that port;
it is passed as that operand, never treated as zero scratch.

Ecosystem maps fresh source RIGHT outputs to the unused caller local slots by
qubit renaming. The SDK's temporary names eliminated by output relocation do not
become idle native scratch. If output bookkeeping aliases an original input,
the shared explicit output-order restoration applies physical SWAPs instead.
When the adjoint returns a consumed local input as a different logical port,
the remaining freed input storage supplies the clean physical local output.
That alias case is checked on every basis column and measurement branch; it
emits six binding SWAPs (four forward and two backward), separate from the
source declaration's bookkeeping costs.

The snapshot allocator also reserves source RIGHT outputs across sibling
operations until their LEFT consumers finish. Eager export can otherwise free
an allocation name before recursive expansion reaches an intervening SELECT.
The previous adapter could borrow a still-live flag as SELECT scratch. The
corrected lifetime handling is in the adapter; no Qualtran source was modified.
Tests compare the complete source-root and composed native streams in order,
including operands, parameters and measurement feedback. Only private wire
identities and overwritten classical-slot versions are consistently renamed;
no gate is reordered or removed. Both complete streams pass every-history
cleanup certificates, including the scalar-local controlled witness.

| Local output | Control | Native reserved wires | Native instructions | Native T/T-dagger, measurements | Independent source T-equivalent, measurements, peak |
| --- | --- | ---: | ---: | --- | --- |
| Scalar | None | 6 | 50 | 8, 2 | 8, 2, 6 |
| Two-element QBit array | None | 7 | 52 | 8, 2 | 8, 2, 7 |
| Scalar | Zero | 8 | 70 | 12, 3 | 12, 3, 8 |
| Two-element QBit array | Zero | 9 | 72 | 12, 3 | 12, 3, 9 |
| Scalar | One | 8 | 68 | 12, 3 | 12, 3, 8 |
| Two-element QBit array | One | 9 | 70 | 12, 3 | 12, 3, 9 |

Both directions retain these figures. All basis columns and a full complex
input vector, with arbitrary logical THRU junk, are checked for every small
measurement history against the unchanged source tensor. The local storage and
private work finish at zero. The zero-selection case executes a RIGHT-output
preparation and its adjoint around a non-self-adjoint SELECT; it does not bypass
the preparation. Symbolic normalization stays metadata with concrete physical
ports. Symbolic typed arrays and all directed junk descriptors retain the source
signature order; unresolved widths are refused before numerical capture.

The abstract preparation dependency is now closed as a complete source file,
[Q-0693 PREPARE base](PREPARE_ORACLE_FILE_AUDIT.md). Its complete member and
documentation map is separate from this consumer's remaining defect.

## Complete source definition disposition

| Source members | Owner and disposition |
| --- | --- |
| `_total_bits`; both classes' size and register properties | Ordered register totals, native layouts and typed signature helpers; concrete, symbolic and shaped declarations verified. |
| SELECT `alpha`, `epsilon`, `signature`, `signal_state` | Native SELECT and its independent PREPARE signal; symbolic norm retained, error remains the source zero placeholder, no PREPARE junk execution port. |
| SELECT `build_composite_bloq` | One existing SELECT emitter; original named/packed factories, both directions and complete history proofs. |
| Both `wire_symbol` methods | `drawer/lcu_block_symbols.py`; empty connector, `B[H]`, and the original literal `control` name rule remain unchanged. |
| LCU `control_registers`, `alpha`, `epsilon`, `signature`, `signal_state` | Optional scalar control, original selection/THRU-junk/target ordering and identity zero signal; typed and symbolic declaration tests. |
| LCU `build_composite_bloq` | Native PREPARE, directional SELECT, original PREPARE adjoint; THRU/RIGHT storage, aliases, zero selection, complete operators and branches verified. LEFT-input extraction fails in the unchanged source. |
| LCU `get_ctrl_system`, `adjoint` | Source-specialized SELECT is captured for each activation value and each direction; PREPARE stays unconditional. Shared control construction remains its own owner. |
| Four factories and two documentation records | Existing named/packed Hubbard catalogue, original tests and independently executed native tutorial; Q-0282/Q-0284 evidence is reused. |

Q-0283 remains open under the frozen closure criteria: the source excludes LEFT
junk from its external signature and then requests that missing input while
building the preparation. Both directions raise `KeyError('local')` in the
reproducible fixture. The adapter preserves this diagnostic and does not invent
an external operand or silently repair the reference. Wire-free scalar carrier
admission remains tracked with the shared Phase wrapper's existing boundary.

Final verification passes 361 concentrated consumer/interop cases, 14
complete PREPARE-base cases, 60 local CPU/representation integration cases,
and the independent LCU notebook kernel, without skips. Strict typing covers
all 169 Algorithms and touched adapter files at the configured Python floor.
Two existing `Self` imports now use the already-used `typing_extensions` owner
so that default check passes; quantum algorithms are unchanged by that fix.
Formatting, ownership, architecture, multi-level IR and public API checks pass.
The
parallel coverage worker failed during Torch import before running tests; it
is not acceptance evidence. Coverage is collected serially with Torch already
loaded, and the ordinary regression runs in two workers. The serial 109-case
coverage subset reports 91% across the six measured modules; changed executable
statements within that scope are 84/86 covered (97.7%). The two uncovered changed
statements reject a non-scalar norm and an inconsistent source register extent.
Coverage is not a substitute for the complete operator and branch comparisons.

```bash
pytest -q tests/unit/test_lcu_sided_junk.py tests/unit/test_prepare_oracle_file.py
```

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_lcu_block_encoding.ipynb
```

## Controlled-Y dependency representations

The complete actual named and packed LCU factory captures are verified against
both installed Cirq profiles. Cirq 1.6 supplies ControlledGate(Y), while Cirq 1.7
supplies CYPowGate. The existing owned importer lowers both to native CY.
Each actual LCU root contains eighteen such operations. Physical instruction
counts are 2490 without control, 2542 with zero control and 2540 with positive
control, thirty-six below historical Sdg/CX/S-expanded expectations. The
standalone SELECT consumer has eighteen CY and 1098 physical instructions,
thirty-six below its historical expectation. Erasure-event, T and clean-work obligations
are independently retained.

The shared erasure certificate accepts CY's Boolean target XOR control update.
An independent full complex-state check verifies its imaginary phase in both
measurement outcomes. Changed erasure controls and dirty returned work remain
rejected. The previous snapshot implementation reproduces the earlier CY
certificate failure, separating this pre-existing verification issue from the
field-factory output-alias repair. Original QEC declarations remain separate.
