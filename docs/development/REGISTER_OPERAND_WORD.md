# Register operand words

K-0261 and K-0262 cover the frozen operand conversion implementation
and header at blobs `554a7c8918b8b66817dcf380b1505ec3d5665927` and
`6ebc7a3d797738e65e35800b55253aefe2571139`.

| Source admission | Native compiler binding |
| --- | --- |
| `from_obj` | Accept a typed singleton or a sized iterable. Preserve quantum, Z-classical, Boolean and X-basis tags, including both phase-state spellings. |
| `from_obj_expecting_list` | Require a sized iterable and reject scalar text/bytes. Preserve the supplied sequence order, including reversed and strided views. |
| `from_obj_or_int` | Expand a non-Boolean integer to the requested width; fall through to normal admission for other operands. Reuse the compiler's big-endian unsigned splitter after wrapping an admitted negative offset. |
| Optional stricter target casts | Check for a pure quantum or Z-basis word at the consuming boundary. Mixed X/Z words remain representable before either cast. |

The source stores expanded integers in little-endian order. Native
expansion yields big-endian tuples for the register emitters. The physical
sequence itself is never reversed by binding. Native singleton operands
use integer wire indices and Boolean objects for literals. When an
integer literal and a quantum word are both admitted, use a one-element
sequence for a singleton quantum index. The source distinguishes its
separate Python qubit object and accepts bare 0/1 as Boolean singletons
outside the integer-expansion path.

The range for width `n` is `[0, 2**n)` for unsigned values and
`[-2**n, 2**n)` for offset conversion with wrapping enabled. A value of
`-2**n` encodes zero. Python's integer splitter covers the source's
uint64 and wide byte branches with the same logical bits. Boolean scalars
do not expand into a full literal word.

The source header wraps empty, scalar, borrowed span and allocated
storage with a singleton marker, move-only free ownership and a stride
view. Native conversion freezes the logical operands in an immutable
tuple. The input containers and physical registers remain caller-owned.
Neither conversion emits gates, allocates work wires or declares a
resource model. Native checks lengths against bounded iteration before
returning a word. Source conversion trusts length when writing the
allocated buffer. In the source owning self-move case, move assignment
can release the storage before retaining its span. Both source cases
are recorded by inspection, outside valid input parity.

Rotation uses sequence admission and equality uses singleton/sequence
admission plus integer expansion. The locked source captures mixed tags,
a reversed view, signed range cases and literal addition results at
widths 4, 64 and 65. The native literal addition test exhausts measured
histories, checks clean workspace and compares complete complex
amplitudes through append. The four-bit native addition emits 13 X,
4 CX, 2 CCX, 3 HMR, 2 Z, 1 CZ and 2 NEG at the KMX boundary.
Literal equality and rotation sequence
bindings are compared against the same compiler streams for explicit
typed words. The separately tracked low-level Python array APIs and
value-ID declarations retain their own acceptance entries.

Run `python -m pytest -q tests/unit/test_kickmix_operand_word.py`.
