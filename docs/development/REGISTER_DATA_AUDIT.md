# Physical register data

The complete locked `circuit/register_data.h` file, inventory parent K-0029,
contains a name, an ordered `QubitOrBit` vector, default structural equality and
a diagnostic stream renderer. Its lock is
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.

## Native ownership

Compiler `formats.kmb.KmxRegister` remains the immutable raw physical register
value. Existing private `_RegisterData` columns retain mutable builder ownership.
`formats.kmx_registers.KmxRegister` remains the logical Program declaration
adapter. This batch adds only `diagnostic_text` to the existing raw value.
No second register value or storage representation is introduced.

The diagnostic retains source field order, literal name contents, member order,
duplicates and the trailing separator. Its type label is `KmxRegister`.
The existing KMX escaping and KMB encoding paths retain their behavior.
Native dataclass equality already compares both the literal name and ordered
members. Native values are immutable, and a copied mutable builder owns its
member lists separately.

## Complete source comparison

The probe compiles the unmodified header and its verified scalar implementations.
Eight root and dependency Git blobs are checked against the immutable inventory.
Twenty source values combine empty, ASCII, quote/backslash/newline, UTF-8 and
embedded-NUL names with empty, quantum, classical and duplicated mixed members.
All 400 ordered equality pairs compare directly, along with every diagnostic.
The original source copy-independence observation is retained.

Source `QubitOrBit{}` has an untyped raw-false default sentinel. A register
containing it renders like one containing `BitId(0)` but compares differently.
The probe preserves both words, both renderings and the unequal result.
Native physical register members are explicitly `q` or `b` IDs. Their contract
has no untyped sentinel alternative. Existing `OperandIdentity` retains the
scalar Boolean/classical distinction separately. No register equality parity
is claimed for an untyped member outside that admitted native domain.

Native tests exercise all twenty values through mutable builder replacement,
freeze, copy isolation and exact KMX/KMB round trips. A normal
`circuit.append(op)` CX consumer verifies that replacing register metadata
preserves ordered physical operations and their counts. Its full complex state
and logical two-qubit dimension are checked. Existing register execution,
feedback and erasure evidence is reused through the batch regression.

## Resource accounting

| Quantity | Result |
| --- | --- |
| Register value and diagnostic primitives | 0 |
| Register value workspace and measurements | 0 |
| Normal append consumer actual primitive | 1 CX |
| Metadata replacement effect on ordered raw records | Exact equality |
| Analytical model | None for register values |

This is the existing local single-device path. No numerical work moves into
Compiler. The shortest contributor check is
`tests/unit/test_physical_register_data.py` with the existing KMX mutable-circuit,
circuit-object, register-declaration and KMB suites. Fixture generation lives in
`tests/unit/fixtures/generate_register_data.py` and requires `--source`,
`--inventory` and `--output`. The source root contains `kickmix/src`.
