# Register operand arrays and views

K-0257, K-0258 and K-0259 cover the locked array binding, declaration and
original test file at blobs `dcfef2ae0d48ea1f6e249595209709b3a052f3e6`,
`3396d7c17b353c569cf3155289711e9a3c26cf8d` and
`604aec7ca30eb5b632bbf822ea42a54f57b8fc94`.

Native conversion reuses the shared operand admission and retains the
physical family of each element. The array owns an immutable dense
tuple. An indexed slice keeps that owner and composes its selection by
offset, stride and length. Negative indexing, nested slices and clipped
empty views use Python range indexing. The common-type tag is inherited
for views and recomputed from visible members for dense construction and
concatenation. The source copy helper has the same dense content contract.
Direct dependencies on array and span storage are covered here for the
Python binding's paths. Their other operations retain separate source
entries.

Equality compares visible typed values, disregarding layout and owner
identity. Concatenation copies visible operands in left-to-right order
and resets the dense offset and stride. A slice can outlive the array
object that produced it because it retains the owner. Diagnostic
representations use native type names and preserve tags and visible
order. The existing typed boundary interprets a plain integer as a
quantum wire and a Boolean object as a literal. The locked Python wrapper
interprets bare integer 0 and 1 as Boolean scalars. A string is admitted
as a phase state only inside the operand sequence.

All ten original cases passed against locked Kickmix. The source
fixture verifies inherited common type, offsets and strides on five
layouts and recomputation after concatenation. The native append case
feeds a reversed view into constant rotation. Its three SWAP operations
and ordered operands match the locked generator. All sixteen complex
amplitudes match the expected permutation. The selected rotation has no
measurements or workspace. The array and view operations emit no gates
and declare no analytical model.

Run `python -m pytest -q tests/unit/test_kickmix_operand_array.py`.
