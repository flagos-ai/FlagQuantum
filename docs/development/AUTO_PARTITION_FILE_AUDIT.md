# AutoPartition source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Frozen bookkeeping files are
Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0318 | `auto_partition.py` | `93cddd9cbb1ab309f713316f066f16d93e5fd8b0` | Actual concrete captures and symbolic-width typed calls verified. Approved 116/125 storage paths are now covered; signed-zero and unresolved physical boundaries remain separate. |
| Q-0319 | `auto_partition_test.py` | `adb86ad3fbfabf4f2aab180d3eecc6ffb9daebf5` | Closed for the original runnable scenarios. |
| Q-0328 / Q-0329 | `partition.py` / `partition_test.py` | `8cebbb58095d172f3db564cdfc231537e6214957` / `c6c3236332577115ff6f09e17f340fe79825e1f4` | Reused for packed binding. Whole-file Partition acceptance continues separately. |

The Compiler's layout binds outer groups in declaration order and the
segments within a group from most to least significant. Typed Program
composition reuses Partition and its adjoint, calls the child once, and
keeps unused segments live and unchanged. For left-only binding it omits
repacking and returns the child right ports in declaration order. The
original left-only unused-segment and missing-input validation is retained.
Pure output-only child declarations are outside the admitted source
constructor as well. A partition adds no physical gates or work pool.
Numerical codecs and classical child callbacks remain in Simulation.

The original input-only and ordinary fixtures, both unused segment
placements, and the fully unused `z` word use the same native layout.
The original controlled three-bit exchange binds `a` as `y, ctrl` and
`b` as `x`, with activation `0110`. The tests traverse the unchanged
source Partition and child calls before lowering literal gates. This
retains the source's generic controlled-swap decomposition before the
Cirq export can replace it with another controlled-unitary synthesis.

| Fixture | Physical stream | Independent locked QEC model |
| --- | --- | --- |
| Original one-bit exchanges | Seven T/T-dagger, no measurements or work in either direction | One CSWAP, logical peak three or six, including unused segments |
| Original 10-port exchange, forward | 57 T/T-dagger, nine measurements, three clean work bits | Three CSWAP, nine And, 33 Clifford, nine measurements, peak 13 |
| Original 10-port exchange, adjoint | 33 T/T-dagger, three measurements, same three work bits | Three CSWAP, three And, 11 Clifford, three measurements, peak 13 |

The forward generic Controlled(Swap) invokes one predicate per bit pair.
The locked adjoint declares ControlledViaAnd around the register swap and
computes its predicate once. The compiler supports both emission paths
by reusing the same control and swap primitives. The complete coherent
state agrees with the original classical permutation and exact source
phase in every forward or reverse history. All work bits are zero at the
boundary. Prior typed Program cases independently establish equivalent
direct calls, reordered multi-dimensional ports, workspace ownership and
measured Add feedback.

The original AutoPartition and Partition non-notebook suite has 20
passes, ten mode skips and one expected failure. The Partition notebooks
listed in its test file belong to the next dependency audit. AutoPartition
has no source notebook. Physical execution admits concrete typed Program
ports. The native declaration retains outer dtype, shape and direction in
ordinary mode. Left-only mode declares the packed inputs on the left and
the typed child outputs on the right in child signature order. Concrete
and symbolic widths, including a symbolic word plus an unused bit, agree
with the locked signature. Ordinary left-only outer metadata can be
declared. The locked decomposition attempts to return such a packed port
and raises a finalization error. Concrete native repacking requires
through, scalar packed ports.

Concrete typed unsigned ports also bind through the existing compiler
path. The three-bit BitwiseNot fixture emits the same three X gates and
operands as the locked decomposition in both ordinary and left-only
mode. The ordinary fixture retains a fourth unused bit. Both full
complex states agree through append. Partition adds no gates or work.
The independent source logical peaks are four and three respectively.
The source and native typed programs now both retain the symbolic
Partition–child–Partition graph, as detailed below. Actual foreign factories
retain their original serialization failure before native resolution.

## Actual source wrappers and measured-child captures

`test_auto_partition_source_binding.py` captures both actual source example
factories through the existing named emitter. Their DocSpec, child title,
independent directions, native JSON, reversed packed ports and coherent unused
slices match complete source matrices through ordinary `circuit.append(op)`.
The factories need no work or measurements.

The original examples explicitly use generic Controlled(Swap). Their forward
capture emits three Toffoli atoms. The source adjoint specializes that child
and emits the seventeen-primitive Fredkin decomposition. Both directional
streams are asserted independently, including operand order, and the round
trip restores every complex amplitude. The first witness using seventeen
primitives for both directions exposed this source asymmetry and was corrected
to the unmodified source paths. The independent source model reports one CSwap
in both directions, with logical peaks three and six. Primitive counts and
modeled categories therefore remain separate.

An actual AutoPartition(Add(QUInt(2), QUInt(3))) wraps the child in reversed
b/a packed order plus an unused bit. Native layout binding and restoration
preserve exact slice ownership. Both independently captured directions retain
the source instruction and feedback order. All four measured histories per
direction agree with the full unnormalized complex source action. All sixteen
history pairs per direction restore the entangled input, spectator and idle
slice, with both work bits zero. Native append-to-IR and JSON retain every
instruction. Each direction emits 20 CX, 6 H, 4 T, 4 Tdg, 2 S, 2 measurements,
2 conditional CZ and 2 resets. Existing Add and packed-child models remain
independent of that executed inventory.

The batch passed 414 concentrated cases without warnings or skips. Formatting,
lint, strict typing for the changed Python file, Compiler scope, architecture,
multilevel IR and protected API checks passed. Frozen source blobs remain
unchanged. Logs are retained under `ftqc-reference-env/g2-auto-partition-binding-*.log`.

Q-0318's concrete external-object capture and symbolic typed decomposition
are covered. Full executable dtype admission, including pending Proposal 116,
remains open. The source's original outer-left finalization failure stays
separate. No parent closure is inferred from finite through-port captures.


## Symbolic-width stored decomposition and source transport boundaries

The existing typed AutoPartition owner now retains finite-port, symbolic-width
Partition/child/Unpartition calls in Core Program. Concrete partition bodies
still use the shared Split/Join builder. An unresolved-width partition is a
typed declaration with the original directional signature and no invented
physical body. The root retains the child call and every producer/consumer
connection, including unused slices. Ordinary and left-only roots with n and
2*n+1 widths match every source operation signature, predecessor and output
position. The child is the existing native Always wrapper around a declared
BitwiseNot terminal. No identity body replaces its unknown physical action.

Standard Program QEC queries with explicit independent terminal estimates
match the complete source model: n or 2*n+1 Clifford operations and zero
partition categories. This is logical model evidence. Unresolved physical
binding still fails because concrete widths and executable bodies are required.
The existing actual measured Add and swap captures supply full complex,
branch, work-cleanup and primitive evidence for concrete execution separately.
No physical primitive, measurement or workspace is added by this extension.

Both actual original factories fail in original `bloqs_to_proto` at depths
0, 1 and 2 with NumPy's inhomogeneous-sequence ValueError before native input.
The six failures are asserted against the unchanged source. Actual executable
factory captures above remain valid and use the existing SDK boundary.
Foreign serialization is not inferred from those successful captures.

Expanded regression exposed one inherited assertion in the direct Partition
test. Replacing the AutoPartition owner with its batch-base source reproduces
the same failure. Forward construction checks executable part descriptors,
whereas inverse logical Split/Join construction preserves signed fixed-point
metadata and succeeds. Physical binding still rejects that descriptor under
pending Proposal 116. The test now asserts each actual boundary. Production
Partition and Core admission owners are unchanged.

Q-0318 remains open for full executable dtype admission, including pending
Proposal 116 signed/empty fixed-point words. Ordinary outer LEFT metadata
continues to hit the original source finalization failure; native scalar
repacking keeps its explicit admission check. Symbolic array dimensions remain
the existing finite-indexed-port boundary. This batch closes symbolic word
width decomposition only, without treating those unresolved boundaries as done.

Validation is recorded in `g2-auto-symbolic-*`: 380 concentrated, 17 integration
and 66 Cirq 1.6.1 cases pass. The two changed production/new-test files pass
strict typing, formatting, Ruff, Compiler scope, architecture, IR and protected
API checks. The retained direct Partition test is checked separately at its
existing annotation boundary. Source hashes and unchanged physical execution
owners are verified in the evidence receipt.

## Whole typed-array views, 2026-11-21

The existing AutoPartition owner also passes whole-array source handoffs for
seven admitted three-bit descriptors, unsigned, bounded, Montgomery, unsigned
fixed point with fractions -2 and 3, GF(8), and degree-two GF(2) polynomials.
Shapes (2,), (2, 2) and (0, 2) run in ordinary and left-only mode. The source
and native roots independently wrap an entire row-major child array in one
packed scalar word. Root dependencies, original emitted operands and physical
aliases agree. Left-only views consume the packed input and expose the child's
original word outputs. The empty array executes with its real control wire.

Full complex appended execution includes an entangled spectator. No private
work, workspace events or measurement feedback are introduced. Actual atomic
CH count equals the number of words. Independently queried original QEC costs
are four Clifford and two rotations per word, with peak 1 + 3 * word_count.
Partition views themselves contribute zero gates. This is part of the 232-case
shaped batch documented in SPECIALIZED_CONTROL_FILE_AUDIT.md and the shared
g2-specialized-shaped evidence receipt. Q-0318 remains open for its existing
signed/empty fixed-point admission and unresolved physical boundaries.

## Complete catalog wrapper dependency graphs, 2026-11-22

The complete AutoPartition and Always catalog dependency group now reuses the
existing Compiler partition, controlled Swap, and And Programs. All eleven
original factory contexts retain their order, signatures, parameters, and
fresh interface drawings. The two AutoPartition roots preserve the original
five/seven ordered calls, including unused slices. Always retains its one
ordered And call. The three source-success roots yield five exact recursive
graphs and twenty-one original atomic/refusal contexts. Both controlled Swap
descendants are qualified by complete structural matching. Changed primitive
operands reject the labels. Fresh comparable source drawings agree.

The published catalog adds fifty-one contextual children. Twenty-eight are
explicit native-only Split/Join rebinding helpers, ten and sixteen beneath
the two AutoPartition roots and two beneath Always. They carry native interface
presentation without a source decomposition-equality claim. Eight other
original factory roots retain their original atomic decomposition refusal.

Selected catalog Programs bind through the existing register binder and execute
with circuit.append(op). Full complex isometries match unchanged original
tensor contractions for all three factories. AutoPartition classical action
is NotImplemented, so its tensor contraction is the executable reference.
Always's RIGHT And target remains a logical output. Existing accepted tests
rerun all applicable measured Add/And cleanup histories, exact recovery,
entangled spectators, noncontiguous bindings, and clean temporary workspace.

Actual AutoPartition primitive histograms are 8 CX, 2 H, 4 T, and 3 Tdg for
each root, with peak widths three/six and no allocation events. The independent
original models each declare one controlled Swap, with seven T-equivalents.
Always emits 6 CX, 2 H, 2 T, 2 Tdg, and 1 S, with one allocated logical target
and peak width three. The original Always(And()) root QEC query fails because
its direct graph delegates to atomic And. This refusal is retained. The
independent original And child model declares one And and four T-equivalents.
All three forward Programs have zero measurements. Primitive histograms and
macro estimates are separate evidence.

Receipts use g2-bookkeeping-ui-* under the external reference environment.
Eight source blobs remain equal to the frozen manifest. Qualtran 0.7.0 and
Kickmix 0.1 Simulator are verified. Production Compiler/Core/Simulation owners
are unchanged. Existing signed/empty fixed-point, symbolic physical admission,
original serialization and generic-source boundaries remain explicit.

The batch verification retains 320 execution/reference passes with five existing
protobuf warnings and 29 local integration passes. Strict typing covers five
changed Python files. Black, Ruff, integration scope, architecture, multilevel
IR and protected API checks pass. The full catalog regression passes 786 cases, with three existing deselections
and fourteen existing notebook escape warnings recorded in the catalog receipt.

## Approved storage dependency closure — 2026-11-25

Whole-array packed roots now cover signed positive-width, unsigned empty and GF(3)/GF(5)/GF(9) targets in ordinary and left-only modes. Original ordered calls, operands, physical aliases, zero work and full complex states match. The signed-target two-control child also matches every source measurement history and an independent clean operator. Signed-zero, original outer-LEFT finalization and symbolic physical boundaries remain separate.

Earlier pending-Proposal-116 refusals above are historical and superseded. See
[the shared implementation audit](NONBINARY_QUANTUM_FIELD_STORAGE.md) for the
approved scope, exact original failure dispositions, regression and resource
evidence. Existing owners and circuit.append(op) are reused. This closes the
approved storage dependency; it does not infer unrestricted dtype or symbolic
physical support or close the remaining whole-file parent boundaries.
