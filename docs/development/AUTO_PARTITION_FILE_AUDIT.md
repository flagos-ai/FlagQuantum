# AutoPartition source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Frozen bookkeeping files are
Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0318 | `auto_partition.py` | `93cddd9cbb1ab309f713316f066f16d93e5fd8b0` | Open for direct external objects and symbolic typed child decomposition. |
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
The source can also construct the symbolic Partition–child–Partition
graph. Direct external objects and symbolic typed decomposition remain
open.

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

Q-0318's concrete external-object capture gap is now covered. Symbolic typed
Program decomposition and arbitrary one-sided packed executable bindings remain
open at their recorded admission owners. The source's original left-only
finalization failure remains separate. No parent closure is inferred from
finite through-port captures.
