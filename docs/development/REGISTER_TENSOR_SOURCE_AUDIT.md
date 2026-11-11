# Register tensor data source audit

Simulation owns word-axis tensor data, numerical projections and active control
subspaces. Compiler still owns program transformation and emitted operations;
Runtime owns execution orchestration. This is a CPU dense inspection path, not
a backend-selection or scalability feature. The unchanged reference is Qualtran
v0.7.0 `simulation/tensor/_tensor_data_manipulation.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Source copyright remains in native
code. Five original Git blobs are verified against the frozen inventory in
workspace `g2-register-tensor-source-audit-source.json`.

## Complete source mapping

| Original definition | Target in `simulation/tensor_network/register_tensor_data.py` |
| --- | --- |
| `tensor_out_inp_shape_from_signature` | `register_word_shapes`: calculate inputs first, return outputs then inputs |
| `tensor_shape_from_signature` | `register_word_shape`: concatenate the two groups |
| `active_space_for_ctrl_spec` | `register_active_indices`: leading control words, same raw integer on both sides |
| `_n_qubits` | `_declared_bits`: sum all declarations' total stored bits |
| `eye_tensor_for_signature` | `register_identity_data`: complex128 identity followed by shared projection |
| `tensor_data_from_unitary_and_signature` | `project_register_tensor`: source assertion, all-THRU signature, reshape, zero selection, result assertion |
| Three original test functions / four parameterized cases | Run unchanged, then with four helper bindings replaced by native adapters |

There are no other functions, classes, constants, exports or assets in the source
implementation/test pair. Existing Core register/signature metadata represents
ports, sides and widths; no foreign register classes enter production. The
control-spec boundary uses explicit native pattern arrays and leading register
metadata, rather than introducing another CtrlSpec type. Original tests convert
metadata only at their test boundary; original files are not edited.

## Interfaces and ordering

Raw shape functions preserve symbolic width powers and source input-before-output
failure order. Symbolic array shapes retain the metadata owner's errors. Dense
shape output is one axis per row-major word element, outputs before inputs.
Bit decomposition below that boundary remains MSB-first. Negative width powers
are not prevalidated at the raw shape entry.

Active control addressing first checks control dtypes for symbolism, then pattern
arrays, preserving the original short circuit. It then builds slices from word
shapes and assigns each pattern value to both output/input control positions.
Integer conversion and NumPy negative indexing are retained. It does not infer
physical wires, apply a signed codec, or enforce the validated caller's count and
shape policy. Symbolic error messages use native guidance instead of an upstream
CtrlSpec repr; metadata errors contain native class names. This is an explicit
representation difference, not literal diagnostic-text parity for those cases.

Projection checks the complete matrix shape, rebuilds all declarations as THRU,
reshapes, and selects output zero for LEFT / input zero for RIGHT. It is neither
a trace nor a reset, and does not verify unitarity. NumPy owns raw dimension
rejection, dtype, subclass, view and scalar behavior. The existing NumPy dependency
is reused, without a new manifest entry, package, registry or numerical core.
Storage remains exponential in declared bits.

The validated `register_tensor.py` entries keep finite-width/array checks and
their native ValueError for matrix arity. They delegate numerical projection and
identity allocation to the raw owner, removing duplicate projection loops.
`controlled_tensor.py` retains child-callback order, THRU/count/shape checks and
scalar refusal, then reuses active indexing; its duplicate slicing loop is gone.
A substituted projection provider and active-index provider prove these consumers
actually use the shared implementation.

A legal native signature can reuse one name on separate LEFT and RIGHT ports.
The validated interface previously supported its numerical projection and keeps
that support: only such paired names are made positionally distinct before raw
projection. Names do not affect numerical axes. Direct raw calls reproduce the
source THRU-reconstruction failure instead. This is a documented adapter boundary,
not a repair to the reference or a new source-parity claim for that native case.
No protected API, serialized contract, Compiler graph or runtime path changes.

## Verification

- 239 new tests cover all 27 three-register direction combinations with seven
  NumPy dtypes, contiguous/transposed/Fortran layouts, views and input immutability;
  zero-width versus empty arrays; symbolic and negative widths; control offsets,
  negative/raw values, empty/multiple patterns and dtype-before-pattern queries.
- All four original test cases pass unchanged and again with only native helper
  bindings plus metadata conversion. Projection additionally compares the
  original implementation directly, including ndarray subclass and scalar
  object behavior.
- Concentrated regression: 585 passes, no skips, three recorded original Autoray
  numerical warnings. It includes component graph contraction, register binding,
  classical tensors and existing append consumers, without new GQSP scenarios.
- The three numerical modules have 134 statements and 46 branches, all covered.
  Strict typing covers these modules and the new audit test. Black, Ruff,
  Simulation scope, architecture, internal IR and API baseline checks pass.
  These are measured scopes, not whole-repository coverage/type certification.
- Default-suite status and prior failure-ID comparison are retained in workspace
  evidence and the live ledger: 7 failed, 21,890 passed, 300 skipped,
  1,037 warnings in 158.97 seconds, stopping at the existing maxfail bound.
  All seven failure IDs match the previous baseline; no new failure is observed.
  This is not a full-green default suite.

The existing controlled-X consumer covers all four two-bit control polarities,
exact source Instructions/operands/feedback, complex amplitudes with an entangled
spectator and both measurement outcomes (eight branch evaluations). Its work bit
returns to zero. Existing graph consumers also recheck their measured-add
forward/roundtrip histories, pure/mixed contractions and gradients. Raw tensor
helpers themselves do not measure or allocate quantum work; no fictitious
measurement branches are attributed to them. Users continue `circuit.append(op)`.

## Separate resource accounts

Each controlled-X consumer emits H3, CX7, T2, Tdg2, S1, measurement1,
conditional CZ1 and reset1, plus X8/X4/X4/X0 for control patterns 00/01/10/11.
Each uses one clean work bit and one reusable classical slot. Its original model
is AND1, Clifford2 and measurement1 (four T-equivalents). Model Clifford categories
are not primitive Clifford instructions, particularly the negative-control Xs.
The primitive stream is compared independently against original decomposition.

Raw word shapes, slicing and projection emit zero gates and use zero physical
work/feedback slots. There is no source QEC cost model for these numerical helpers.
Identity allocation uses complex128 dense storage, 16 * 4**n bytes before
projection for n declared bits; dtype-preserving projection may return a view.
No distributed or hardware performance claim is made.

## Inherited open limitations

The source accepts a Signature with same-named separate directional ports but
fails while reconstructing its all-THRU signature. An empty-signature one-by-one
object matrix containing SymPy `I` produces an object scalar without `.shape`,
then fails at the source result assertion with AttributeError. The raw owner
preserves both behaviors. The source projection prose also reverses the selected
LEFT/RIGHT sides; native docs describe the actual code above.

Q-0894 closes for its entire original test file. Q-0893 remains open under the
frozen source-defect rule despite all six function call chains being mapped.
Previously recorded QSP, Jacobi-Anger, permutation and other tensor limitations
remain separate. No new child IDs, parent denominator or G3 work are introduced.
