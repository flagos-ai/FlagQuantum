# Bookkeeping library constructor dependencies

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. This batch maps the complete
`allocate.py`, `free.py`, `split.py`, `join.py` and `cast.py` source constructor
group from the existing 303-entry serialization resolver. Their bodies retain
the existing layout and lifetime owners. No foreign constructor executes in
Compiler and no new Program opcode is introduced.

## Interface, bit order and ownership

`compiler/formats/legacy_bookkeeping_factories.py` supplies an independent
mapping to the existing `resolve_component_program` interface. It composes
with `fixed_gate_factories()` using ordinary dictionary union. The native
resolver still checks serialized signatures, connected ports and complete
Program linear ownership before physical binding.

| Source class | Native owner | Logical interface |
| --- | --- | --- |
| Allocate | `layout/component_workspace.py` | dtype, binary dirty flag, RIGHT reg |
| Free | `layout/component_workspace.py` | dtype, binary dirty flag, LEFT reg |
| Split | `component_rebindings.py` | LEFT typed word, RIGHT MSB-first reg bit array |
| Join | `component_rebindings.py` | LEFT MSB-first reg bit array, RIGHT typed word |
| Cast | `layout/component_cast.py` | inp_dtype, out_dtype, row-major shape, two directional reg ports |

The serialized integer dirty flag is validated and passed as a native boolean.
Cast shapes retain scalar, vector, multidimensional and empty-array ordering.
Serialized shape arrays are decoded without changing values. Unknown fields,
nonbinary flags, malformed dtype arguments and conflicting signatures fail at
the existing data boundary. Absent signatures derive from the known native
constructor. Runtime storage checks stay with the original native owners.

Clean allocations use the caller's clean pool and must be restored to zero.
Borrowed allocations use the separate dirty pool and must retain their full
joint complex state. Free consumes an owned lease. Freeing a caller input and
returning a live allocation each require their existing explicit ownership
transfer options. The mapping does not infer either transfer from a signature.

## End-to-end execution and resource accounts

The executable scenario in `tests/unit/test_legacy_bookkeeping_factories.py`
serializes an original composite that allocates a three-bit word, splits it,
applies two H gates per element, joins it, casts the word and frees it. The
native Program resolves both constructor groups, binds the reversed work pool,
and supplies the existing `FieldOperation` to `circuit.append(op)`.

The source and native dependency-comparable gate streams and ordered physical
allocation/free events agree. All complex spectator amplitudes return under
both clean and borrowed ownership. The clean pool returns to zero. The dirty
pool may begin entangled with the spectator and preserves that joint state.
The component group has no measurement branches.

| Account | Observed result |
| --- | --- |
| Bookkeeping physical primitive stream | Zero gates, resets and measurements |
| Independent source bookkeeping QEC model | Zero in every category |
| Complete execution fixture | Six H, zero measurements and three reused work wires |
| Independent complete fixture QEC model | Six Clifford and zero T/measurement |

Existing classical, tensor, control, drawing and foreign callback evidence is
reused from `ALLOCATE_FREE_FILE_AUDIT.md`, `SPLIT_JOIN_FILE_AUDIT.md` and
`CAST_FILE_AUDIT.md` and their accepted complete source tests.

The new constructor scenarios pass 51 focused cases. The complete affected
native regression passes 343 cases without skips, including the existing
foreign factory, classical/type, ownership, symbolic lifetime and source
measured-erasure consumers. The local integration selection passes 145 cases
with 18 existing emulation warnings. The new owner has 100% statement and
branch coverage. Strict typing passes both changed Python files. Black, Ruff,
Compiler scope, language, architecture, IR and API checks pass. Dependency
policy reports the identical existing 720 findings. Frozen source and original
test fingerprints and the locked Kickmix 0.1 Simulator are verified separately.

## Explicit remaining boundaries

The native mapping preserves current storage admission. Symbolic Allocate and
Free remain valid logical declarations and refuse unresolved physical width.
Split and Join require concrete widths. Cast execution requires concrete
nonnegative shapes and existing executable storage. Quantum/classical Cast
continues to fail at its existing owner even if the foreign permission flag
is true. This batch does not authorize protected storage expansion.

Source dirty tensor contraction still projects onto zero, independently of its
borrowed physical lease convention. Source Cast adjoint shape/permission loss
and shaped tensor limitations retain the dispositions in `CAST_FILE_AUDIT.md`.
The source boolean-to-int protobuf deprecation remains visible. Complete Q-0876
stays open. Combined built-in mappings cover 13 identifiers, leaving 290 other
catalogue identifiers and generic reflection for subsequent source groups.
