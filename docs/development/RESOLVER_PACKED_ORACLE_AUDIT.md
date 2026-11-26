# Packed PREPARE and SELECT constructor disposition

Reference: complete Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389` source/test pairs:

| Source | Git blob |
| --- | --- |
| state_preparation/black_box_prepare.py | 42aacba78eb58628d93eee0db860754827d2f42e |
| state_preparation/black_box_prepare_test.py | 8dc0acab59408715563bfd651249bc516162fea5 |
| multiplexers/black_box_select.py | 8cb8d402b17f17952aa3f06b3fca8ed4cc048239 |
| multiplexers/black_box_select_test.py | 6bd3dd68d3adef743a4ae6990180951c40d928d0 |

## Ownership and replacement

Compiler layout owns the shared packed oracle graph, retaining one AutoPartition
call and its exact source Partition/callee/reverse-Partition dependencies.
The existing Algorithms packed_prepare_program entry delegates to that owner.
Its signature, defaults, zero-selection behavior and declared-child checks
remain intact. The duplicate graph construction was removed. Compiler receives
only Core Signature/Register metadata and complete native Program children;
Algorithms types and SDK objects do not cross that boundary.

Formats requires explicit selection/junk or selection/target metadata and exact
child inverse Programs from the owning provider. Compiler legacy_hubbard_packed_context supplies the original
Hubbard example Programs through existing native owners, while Ecosystem
injects numerical coefficient conversion. Additional native
owners can replace that dependency without modifying the packed consumer.
Unknown roles are not inferred from names, widths, amplitudes or gate counts.
Normal consumption remains circuit.append(op).

Packed words are MSB first; shaped child elements concatenate in row-major
order. Selection, logical junk and system storage remain caller-owned. Only
child scratch is private. The zero-selection PREPARE source root remains an
identity, including with nonempty junk. SELECT retains selection/system words
and the original omitted-control boundary. A non-self-adjoint, array-target
SELECT fixture proves exact inverse ownership, complete complex matrices,
recursive calls and zero wrapper work.

## Direct AutoPartition dependency repair

The existing AutoPartition builder required a call-only child body even though
it never inspected the body to construct its one typed call. A valid primitive
FunctionRecord was therefore refused. The builder now validates the complete
Program, typed entry and physical-context boundary without imposing that extra
call-only requirement. Bodyless declarations retain their existing refusal at
physical binding. Existing layout, type, side, name and output checks remain.
An unchanged original CNOT child with an unused bystander slice matches source
partition calls and all complex matrix columns. No IR opcode, protected type
admission or backend contract changed.

## Source and numerical evidence

Both original examples resolve from depth-zero source libraries in both
directions and retain exact root signatures/calls. The full original packed
Hubbard PREPARE and its recursively comparable native dependencies match source
calls and operands. Bound primitives and the complete allocation/release trace
are identical to the already accepted Algorithms entry in both directions.

The unchanged stored Hubbard and packed tests are rerun rather than copied.
They include coherent logical-junk states, every small raw measurement history,
private scratch zero and independent exact inverse execution. Existing numerical
witnesses fill the source's atomic ModAddK leaf using its classical action;
this is not claimed as an upstream primitive decomposition. The source atomic
boundary and independent resource model remain explicit.

Source-free constructor resolution, SDK-free physical binding and append retain
identical emitted payloads. A native owner for the non-Hermitian SELECT fixture
replaces the Hubbard dependency through the same role contract and preserves
array descriptors and original phases.

All four original blobs, Qualtran 0.7.0 and actual Kickmix 0.1 Simulator are
verified in the batch resource receipt. Unchanged original tests have eight
passes, two source count skips (counts only available from decomposition), and
two original serialization xfails. Original depth-one serialization independently
fails on heterogeneous AutoPartition arguments with the NumPy ragged-array
ValueError. The reference and its arguments were not modified.

## Resources and boundaries

Packing and partition views add zero primitive gates, measurements or work.
The original Hubbard PREPARE emits 47 T-family gates, 66 CX and three measurements
with two peak private leases, ten logical input wires and twelve live peak
wires. Its independent source model charges five CSwap, two And, eleven Clifford,
four rotations, two measurements and twelve logical peak qubits. Native CSwap
lowering and atomic ModAddK descendants remain separate from that model; no
primitive/model equality is asserted below incomparable leaves. The supplied
18-wire pool is not reported as the measured live peak. Both directions return
zero live private leases; numerical zero follows from the reused branch tests,
not lease release alone.

The former uncontrolled Hubbard SELECT bodyless gap is now closed through the
existing private scalar carrier and the shared native SELECT body. Both packed
directions have the same primitive stream and physical lease offsets as the
direct source-shaped child. See the
[PREPARE constructor closure](HUBBARD_PREPARATION_CONSTRUCTOR_AUDIT.md) for the
active dependency replacement. Proposal 119's new opcode remains pending and
is not used. Controlled Hubbard SELECT serializes at depth
zero but its original packed decomposition fails because control is absent from
the partitions; native construction retains the same failure. Source control
value zero is independently refused by the original constructor.

Other child factories still require their native owner; singleton Hubbard,
symbolic physical weights, one-sided unsupported layouts,
pending protected proposals, wider generic catalogue and repository/GPU/performance
acceptance remain separately scoped. No generic catalogue increment or complete
walk/LP scalar closure is claimed from this constructor group.

Local acceptance: 118 concentrated passes, 145 local integration passes and
34 affected tests after annotation/optional-dependency guard cleanup. Strict
typing covers all seven changed Python files. Black/Ruff, all three team
scopes, repository language, architecture, multilevel IR, public API and diff
checks pass. Dependency policy retains the identical 720-finding baseline;
the new optional NumPy reference test is guarded rather than accepting a new
finding. Existing test callback annotations and a shadowed local name were
corrected without changing assertions. No reference file or protected API
snapshot was changed. Receipts use g2-packed-oracle-constructor-*-20261125.

## Shared scoped context continuation

The original packed/Hubbard role, inverse and specialized-control context now
belongs to Compiler. Ecosystem retains coefficient conversion and the existing
entry signatures. Algorithms object construction is removed from pure SELECT
parsing; both native signatures and literal Programs reuse existing Compiler
owners. See [the complete dependency audit](HUBBARD_PACKED_CONTEXT_OWNER.md).
All 22 original Program/binding/failure records remain identical; source enum
serialization and singleton refusals remain explicit.
