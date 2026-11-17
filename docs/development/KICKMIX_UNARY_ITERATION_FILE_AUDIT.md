# Kickmix unary iteration parity

This audit covers K-0105 through K-0108 at Kickmix revision
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.

| Source file | Git blob | Native coverage |
| --- | --- | --- |
| `gen_unary_iteration.cc` | `ec97900dbce6ac31ca66a33eb63e95ea7073fb1d` | Cursor movement, measurement cleanup and callback traversal |
| `gen_unary_iteration.h` | `d22bf77ef10a465fd75e218a0077fdf6a162451a` | Address, control, work and result-slot contracts |
| `gen_unary_iteration.perf.cc` | `9eb9c4b97a518961d83a886fe96983554a3b0791` | 10-bit controlled sweep over all 1,024 addresses |
| `gen_unary_iteration.test.cc` | `8bac49bb20e02d48d279573a773da0502d89fefb` | Unchanged original suite and native marked streams and branch witnesses |

The native address is big-endian and accepts at most 63 wires. Visits are
unsigned 64-bit values. The caller supplies `max(1, address_width)` clean
wires; the first is the match qubit. Cursor construction reserves work and
emits no gates. The cursor parks at the address-space size after an explicit
`close()`, which clears the owned work and is idempotent. No destructor
emits gates. While a cursor is live the callback must preserve the address,
control and cursor work. An explicit classical guard covers the callback and
the cursor, and its slot is distinct from the reusable HMR result slot.

The physical export reuses the existing cursor and records each nonempty
transition's source mark. Duplicate visits and parked-to-parked moves emit
no marks. Callbacks still run for duplicates and parked visits. Empty
lists do not construct a cursor. Register directives are excluded from the
stream digests; every remaining opcode, ordered operand, condition and
mark is retained. The existing fixture covers 90 streams up to 63 address
bits. The additional oracle covers 73 complete marked streams, including
combinations of quantum and classical control and the full sweep. The
committed fixture was reproduced byte for byte with the unmodified source.

The 14 original tests, 32 native focused tests and 5,276 concentrated
regression tests passed. Native branch tests exhaust small reset and
measurement histories, including both values of the classical guard with a
coherent quantum control. They compare unnormalized complex amplitudes
over the full state, including the zero-work subspace, and reuse the existing
`Circuit.append` example. Format, strict typing, architecture, internal IR
and public API checks passed.

## Controlled sweep

The exact 10-bit workload uses 10 clean wires and one reused measurement
bit. It emits 7,161 operations, plus 2,050 serialized block marks.

| X | Reset | CCX | CX | HMR | CZ | Mark records |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,046 | 1,023 | 1,023 | 1,023 | 1,023 | 1,023 | 2,050 |

These are target opcodes. The source's Toffoli estimates for the 73-value
partial sweep are separately asserted by its tests (78 with quantum control,
77 without or with only a classical guard). Native execution expands reset
and HMR into dynamic instructions; those expanded counts are not included in
the table.

The locked C++ workload took 88 μs in the local harness. The native Python
median was 10.828 ms across seven generation runs after one warmup, while
local checks were running. Both time transition generation and close; the
native figure also includes physical artifact validation. Neither measures
simulation time.

Regenerate the marked fixture from the locked tree:

```sh
python tests/unit/fixtures/generate_kickmix_unary_iteration_file.py \
  --source-root <locked-kickmix-root> \
  --output tests/unit/fixtures/kickmix_unary_iteration_file.json
```
