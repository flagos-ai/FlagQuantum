# Sparse chemistry walk and tutorial audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/sparse`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0412 | `sparse.ipynb` | `369ce50966f7d848ffca9be2caa7b9f04db96fc5` |
| Q-0413 | `sparse_test.py` | `4b0c04ec18ef9abf0aef050231f3147b27bb078f` |
| Q-0414 | `walk_operator.py` | `54d0a3574e81e2f38a71c4de716beaccd2a47bba` |

The original factory takes modified one-electron integrals and
chemists' two-electron integrals. The native factory reuses the
same data owner and verifies default blocking and LCU normalization.
It accepts the same alias and rotation precisions, and provides
the existing optional threshold and explicit blocking controls.
The source forms an unordered union of PREPARE and SELECT
selectors. The native declaration uses deterministic name order
and matches each source register by name, dtype, shape, and side.
The physical layout keeps all PREPARE outputs as signal ports.

The two original tutorial fixtures preserve the seeded six-spin
PREPARE integral arrays and the four-spin SELECT. The call graphs
include each direct original child with its multiplicity and
resource budget. This includes the negative and positive sign
controls and distinct controlled and free Majorana calls. The
native kernel verifies both signatures and both call graphs without
importing upstream. The original and native tutorials each have
18 cells. The original OpenFermion 1.6.1 suite returned three passes,
with its notebook test handled independently. Native numerical
tests on the same 8-, 12-, and 16-spin fixtures match all three
generalized CCZ component budgets separately.

## Physical and source resource boundaries

The native operation uses `circuit.append(op)`, with SELECT followed
by the reflection about the complete signal. The adjoint reverses
the order and conjugates SELECT. The two-orbital unblocked and
blocked cases compare full complex states and work cleanup, with
both values of every feedback slot represented. The unblocked
case emits 278 T/T† and 38 measurements; the blocked case emits
418 T/T† and 47 measurements. The corresponding declared source
budgets are 152 and 196 T equivalents at peaks of 34 and 47 logical
qubits. Physical emitted counts use a separate scope.

The native one-orbital, two-row unblocked walk now composes the new
PREPARE and SELECT paths. Both directions emit 74 T/T† and 15
measurements with 14 reusable work wires. The tests compare the
complex walk and adjoint to the independent Majorana and projected
reflection reference under four representative histories. Matching
support sets prove that every scratch wire returns to zero.

The same one-orbital walk now supports the two-row table in a complete
two-cell block. It retains ten lookup bank wires as part of the signal.
Both directions emit 134 T/T† and 20 measurements using 19 reusable work
wires and 18 classical slots. Forty wires carry instructions. The test
also binds a coherent spectator. Its state is entangled with the system
through distinct complex coefficients across all eight system/spectator
inputs. Both directions match the independent full-signal reflection
under four representative histories. The test compares every nonzero
amplitude and checks scratch zero and the spectator's absence from
all instructions. The same strengthened state test covers the unblocked
case.

The source reflection retains its narrower mark and allocates
fresh QROAM junk. A test that replaces the opaque original erase
leaf with the coherent inverse still observes 3/64 or 1/32
probability of nonzero junk, for the unblocked and blocked
fixtures respectively. The native complete-signal reflection
avoids that failure. Source QROAM erase and the source one-orbital
aggregate model remain unavailable.

The concentrated regression passed 132 cases. Strict typing,
format, ownership, architecture, IR, and public API checks passed.
Q-0412 and Q-0413 close the original tutorial and tests. Q-0414
retains the full source-to-physical reflection and erase boundary.


## Complete factory and dependency closure, 2026-11-28

Q-0414 closes the complete `get_walk_operator_for_sparse_chem_ham` file and its
direct PREPARE, SELECT, SelectBlockEncoding, reflection and Walk dependencies.
Integral shape assertions, integral normalization, source default blocking and
both precision arguments are accounted for. The literal source context retains
the source narrow reflection projector and all required `signal_` environment
ports. It does not substitute the distinct full-signal mathematical reflection
used by the earlier Algorithms operation.

The subsequent [bank-unitary owner](SPARSE_WALK_BANK_UNITARY_AUDIT.md) supplies
exact arbitrary-bank load/unload for this composition, without pretending that
an isometry's measured matching-image erasure is its general inverse. Small
ordinary and positive-controlled contexts compare original block calls and
operands, full complex joint states, exact reverse and private work cleanup.
Existing all-history local certificates and small bank histories are reused;
full Walk histories and the large dense operator are not exhaustively evaluated.
The actual original eight-spin, sixty-five-row, twelve-keep-bit factory binds
and counts both directions with the extra source environment retained. Rejecting
the narrower old native ABI is intentional; omitting those banks is not fidelity.

The source bystander SELECT control, narrow reflection's changed-junk behavior,
opaque erase leaf and generic controlled-adjoint model failure stay explicit.
Under the approved faithful-migration rule these source defects do not keep an
implemented complete file open. The independent native bank extension is never
reported as an existing upstream primitive recipe. The earlier open conclusion
above is superseded for Q-0414. Generic Q-0658 remains open for the actual large
THC dependency context, not for this now-delivered sparse context.
