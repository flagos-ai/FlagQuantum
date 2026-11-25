# Split and Join source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Frozen bookkeeping files are
Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0325 | `join.py` | `b194add90de87b8aeeb05dd662747701394f400a` | Approved Proposals 116/125 storage paths verified; signed-zero Proposal 124 remains separate. |
| Q-0326 | `join_test.py` | `8cd06b0d322692c18bd66b659bf85b9b76ddb3cf` | Closed for the runnable original scenarios. |
| Q-0330 | `split.ipynb` | `f7150b192af058a668f5efb0d599fdbe03f658bd` | Closed by the complete native Split and Join tutorial. |
| Q-0331 | `split.py` | `ab02e2e6f41402c732a9f30635e83227a09d2c4a` | Approved Proposals 116/125 storage paths verified; signed-zero Proposal 124 remains separate. |
| Q-0333 | `split_test.py` | `8efb0a620f8939ee3c4d494f519afdd2a707e284` | Closed for the runnable original scenarios and notebook. |

The Compiler's existing typed rebinding consumes each owned input exactly
once. The word and bit array both use `reg`, on opposite sides. Split
preserves the physical order and exposes an array of one-qubit words
in big-endian order. Join uses the order of the supplied bits and
declares the output dtype. An output permutation is a wire alias
change. No SWAP gates, workspace, or measurement are emitted by either
operation. The typed ports match the source at widths five and 123,
among the tested widths. Both models report zero gates and logical
peak equal to the declared width.

The original ten-wire Allocate–Split–Join case transfers the live
output lease with all physical wires in caller order. The full 1,024
component complex output is the source zero state. The original
four-wire example flips the second logical bit and returns the value
four. The native circuit emits exactly one X for the child operation.
Existing tests also cover complex spectators, permuted return bindings,
single consumption, and measured downstream callers.

Simulation owns the numerical encode and decode order. Split uses
`np.asarray` without coercing the raw callback result to a fixed NumPy
dtype. The typed bit-array collector separately produces the source
call's `uint8` output. Join passes the list of bits to the declared decoder. The original
255 and overflow cases match for Python integers and NumPy scalars.
The original suite covers NumPy's version-specific scalar-constructor
behavior. Empty wire views are valid, while the original zero-width
integer paths fail. Symbolic Split and Join are refused by the
original dtype constructors.

The unchanged original non-notebook run has twelve passes and six
fixture-mode skips. The original 14-cell and native 16-cell notebooks
passed independent kernels. The native tutorial preserves both
four-bit signatures and demonstrates zero-gate bindings around the
original one-X example through append.

The Drawer callback uses a single space on the packed word and index-only
labels on the bit array. Full typed score geometry matches the locked
source for a single bit, a two-bit finite-field word and a five-bit
unsigned word in both directions. A zero-width Join also matches. The
locked zero-width Split score raises `KeyError: RightDangle` with no output
elements. Its scalar callback still agrees with the native score glyph.

All four concrete types preserve wire order through both native
rebindings and emit zero gates or work events. The zero-width rebindings
are valid inside a physical circuit, verified with an untouched spectator.
The physical binder rejects an entirely wireless circuit. The source
Cirq callback can return an empty word. The locked resource model reports
zero gates and logical peaks of zero, one, two and five respectively.
Both source constructors and native executable builders reject symbolic
widths. Complete physical dtype admission remains open. The direct external
consumers verified below close the concrete foreign-factory obligation.


## Actual foreign factories and admitted storage families

The actual `_split` and `_join` examples and eight other concrete dtype cases
round-trip the original deterministic foreign libraries byte-for-byte at depths
zero, one and two. Explicit native factories resolve their canonical nominal
signatures into the existing Split/Join bodies. Bit, opaque, unsigned, signed,
unsigned fixed-point with negative fractional width, ones-complement, bounded,
Montgomery and empty opaque storage retain reversed caller wire order. The
original direct Cirq reshapes and PennyLane no-operation callback agree.
Binary-field and field-polynomial bodies also agree with the direct source SDK
and tensor action. Their original serializer raises the same unrecognized-dtype
TypeError at all three depths, retained separately from physical execution.

Every small nonempty case compares the full complex tensor action with a coherent
spectator through ordinary `Circuit.append(op)`. Empty opaque storage preserves
the spectator. All paths emit zero primitive gates, feedback, measurements and
work events. Independent source QEC cost is zero and logical peak is the dtype
width. Existing downstream measured arithmetic evidence is reused.

Two inherited symbolic-refusal assertions fail on the unchanged batch base
959296a56 because logical-width validation now runs before physical descriptor
admission. They now assert the existing concrete-logical-width failure. The
source constructor still rejects symbolic dtype widths. No production admission
or reference code changes are included in this batch.

Batch evidence `g2-partition-foreign-*` records 776 concentrated, 17 local
integration and 122 Cirq 1.6.1 passes, strict checks on three changed test files,
formatting, scope, architecture, IR and API checks. Full-source parents Q-0325
and Q-0331 remain open for unapproved signed/empty fixed storage under Proposal
116 and any additional unadmitted full dtype families. Concrete foreign examples
do not establish unrestricted dtype support.

## Approved storage dependency closure — 2026-11-25

Both original classes now resolve signed positive-width and unsigned empty fixed-point source libraries at depths zero/one/two. Empty Split -> Join executes in an actual H/T spectator context with exact aliases and full complex operator comparison. Nonbinary field and polynomial direct SDK views also pass, preserving the original codec refusal. Signed-zero storage remains rejected under pending Proposal 124.

Earlier pending-Proposal-116 refusals above are historical and superseded. See
[the shared implementation audit](NONBINARY_QUANTUM_FIELD_STORAGE.md) for the
approved scope, exact original failure dispositions, regression and resource
evidence. Existing owners and circuit.append(op) are reused. This closes the
approved storage dependency; it does not infer unrestricted dtype or symbolic
physical support or close the remaining whole-file parent boundaries.
