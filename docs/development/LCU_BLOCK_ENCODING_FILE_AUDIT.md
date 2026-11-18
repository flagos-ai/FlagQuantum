# SELECT and LCU block encoding source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen files under
`qualtran/bloqs/block_encoding/` are Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0282 | `lcu_block_encoding.ipynb` | `4f457c91bedc3baaf94f7515242a67fffc4bf4b7` | Closed by the native tutorial. |
| Q-0283 | `lcu_block_encoding.py` | `679ac7964baf62288713078a5ec4dbce674cbece` | Named/packed types and source instruction capture verified. Native external oracle-provider replacement remains open. |
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
same physical allocation.

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
configuration. Arbitrary input states, exhaustive long histories, and
foreign oracle object binding remain open.

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
| LCU | 38 | 2526 | 86 | 540 |
| LCU, control on zero | 39 | 2578 | 89 | 552 |
| LCU, control on one | 39 | 2576 | 89 | 552 |

Both directions retain these figures. The source abstract model still reports
74 or 77 measurements and its own logical peak as shown above. The added
physical work and erasures belong to native realization of the declared
arithmetic. An earlier generic realization used more work for unit steps;
the accepted path reuses the same unit-step owner as native Hubbard PREPARE.

These dependencies pass 166 concentrated feedback, modular, source-root,
block-snapshot and sparse-oracle regressions. Strict checks cover the three
production files and shared certificate; format and architecture checks are
recorded with the publication. Q-0283 remains open: instruction capture and
history proofs do not replace consumer-level native PREPARE/SELECT providers,
preserved typed/sided junk ownership, or full source/native state comparisons
through those providers. The existing signal PREPARE must stay nonidentity
for SELECT. Its SDK object must not become native metadata.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_lcu_block_encoding.ipynb
```
