# In-place quantum addition and subtraction

K-0046, K-0047 and K-0048 track the full locked `gen_iadd.cc`,
`gen_iadd.h` and `gen_iadd.test.cc` files (blobs
`f32c7ef12e016bd1b3bc1d511a95cbcb81b5ae85`,
`4f5ec2cb7094c09ad4405456c348ef8b71c52a56` and
`911fe407e1b1d0e1fe423f9e6666126426313369`).

`inplace_addition_ir` owns the shared carry schedule. Its ports are
big-endian, and the implementation walks the source lanes from least
to most significant. Wide offsets first supply the high update, then
lend the unused offset span. Short offsets use caller-cleared padding.
The source chooses clean measured carry or borrowed reversible carry
according to the available workspace and space mode. Subtraction
conjugates the shared adder with X on the target. The algorithm callers
still append their normal operations with pre-cleared workspace.

`physical_inplace_addition` lowers those fixed ports for serialization.
The caller owns register declarations and chooses the optional block
name. The physical form emits allocation resets on leased carry wires.
The subtraction wrapper keeps its own mark and suppresses the nested
adder mark. An empty name suppresses both marks. The physical generation
command uses this same entry for `iadd`. File width is inferred from
declared registers and actual operands, including a borrowed wire
introduced by the lowest-ID three-control fallback.

## Verification

All four unchanged source tests pass, including both six-bit diagrams
and the add and subtract fuzz tests. The checked-in captures match the
two original output diagrams plus the two corresponding subtraction
diagrams, preserving physical wire and operand order, measured feedback,
and exact ASCII layout. The complete KMX statement sequence matches
as well.

A physical-stream comparison exhausts target and offset widths 0–7,
zero through four clean wires, zero or one dirty wire, both control
routes, both space modes, add/subtract, and default/custom/blank tags.
Of 15,360 combinations, 13,224 admitted streams match and 2,136 missing
workspace inputs reject in both implementations. The comparison
normalizes the final newline of the source ostream output. This extends
the earlier accepted 5,104-stream algorithm check to explicit resets
and named physical boundaries.

The physical execution tests enumerate every reset and carry branch
for small superposed registers and arbitrary borrowed values. Each
branch preserves complex amplitudes, the offset and enable, restores
dirty work and returns clean storage to zero. Existing dynamic append
tests cover the wider algorithm paths.

The six-bit controlled diagram with no workspace emits 20 CCX and 34
CX. Three clean wires give 16 CCX, 20 CX, 3 CZ, 3 R and 3 HMR. The
matching subtraction diagrams add 12 X. Register and mark records are
excluded from primitive counts. Algorithm calls with a pre-cleared
clean contract omit the explicit allocation resets. The estimated
CCZ-state budget is reported through the separate analysis consumer.
