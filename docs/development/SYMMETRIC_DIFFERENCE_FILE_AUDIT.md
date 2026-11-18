# Symmetric difference complete source responsibility

Q-0205 and Q-0206 cover the exact frozen implementation and original tests.
The new file audit checks both blobs, all three actual factories, the concrete
DocSpec subset and every ordered signature port, dtype, shape and side.

Compiler `sorting/symmetric_difference_declaration.py` owns standalone metadata
and the original ten-edge concrete call graph. `symmetric_difference.py` owns
the executable set oracle and its clean-work contract. Shared model dependencies
stay with comparison, control, bitonic merge and Hamming components. In-place
sort now reuses the same comparator model. There is no alternate graph registry.

The root declaration retains the source's merge using left length for both
halves, equality on every adjacent, Hamming count and its adjoint, floored
cardinality predicate, word XOR copies and final CNOT. Both directions and
source insertion order remain visible. Individual child QEC queries and root
peak/category queries match source across unequal/equal sets and even/odd
requested cardinalities. Signed/opaque metadata are carried by RegisterType,
while the actual emitter keeps its existing unsigned sorted-set domain.

The actual 4+2, four-bit, four-output factory declares 96 CSwap, 205 logical
And, 2276 Clifford and 205 measurements, with peak 55 logical wires. The native
executable network separately emits 6044 T/TDG, 14953 CX and 797 measurements,
with 113 clean work wires. The smaller 2+2, two-bit, two-output emitter retains
752 T/TDG, 1833 CX, 97 measurements and 37 clean work. These measured accounts
include compaction and parity checking absent from the original root model.

Existing tests retain exact comparable dependency streams, operands and
feedback, exhaustive small sorted-set pairs and occupied outputs, complete
complex amplitudes with spectators, inverse recovery, all applicable small raw
histories and work cleanup. No emitter changed in this batch.

The original non-notebook test file ran unchanged with 6 passed, 9 skipped,
2 xfailed and 1 notebook deselected. Its existing unknown-notebook-marker warning
is retained. Source autotester skips and xfails do not constitute native
executable acceptance.

The source has no root decomposition. Its register prose calls flag a duplicate
flag, while its class and cost predicate specify cardinality matching. Its model
has no impossible-parity rejection or compaction. Native generic symbolic QEC
queries, source diagnostic shapes outside the positive power-of-two merge domain,
foreign/static execution admission and original test fixture gaps remain open
under Q-0205/Q-0206. Standalone symbolic signatures and source symbolic factory
test evidence do not claim a native symbolic cost implementation.

For 1+1 one-bit inputs and zero output words, the original root's QubitCount
fails through its one-bit Hamming child. Its QEC query still matches the native
model. Native explicit-storage accounting returns five peak wires, separately
from that failed source query. This boundary is checked directly.

The final concentrated batch passed 820 cases without skips. One existing
IPython temporary-directory fallback warning remains. Changed production/tests
and all extracted notebook code passed strict typing, Black, Ruff, Compiler
scope, architecture, multilevel IR and protected API checks. Four notebook
figures were rendered and visually inspected.
