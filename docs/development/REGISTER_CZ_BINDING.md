# Native CZ binding

K-0203 through K-0206 cover the four locked CZ files.

| File | Frozen Git blob |
| --- | --- |
| `cz.cc` | `e418d2ebe013f419f2ac5a8128b5431f3e198dd7` |
| `cz.pybind.cc` | `0ec868514492881fb496ae4270f1654bed8f7846` |
| `cz.pybind.h` | `0444390a5a69b49dfe4abfc59f29ed5a31424d8f` |
| `cz_test.py` | `bbe549772ebd0dd11179c33df360dc6f44e15d15` |

The shared width resolver feeds the native Z-control emitter. Both
operand positions accept a quantum wire, classical slot or Boolean
literal. The homogeneous source paths and mixed fallback emit the same
ordered effective stream as native lane dispatch. Two quantum inputs
preserve control order. A classical or literal input narrows the gate
to guarded or unconditional Z or NEG. Repeated scalar negatives stay
separate in the emitted stream, even when the overall sign is positive.
An explicit zero-width layout emits nothing. Nonempty X-basis controls
are refused before append.

The 16 original scalar and 81 original broadcast layouts match effective
source gate and guard digests. Source scope pairs and native predicates
express the same conjunction. Duplicate classical guards reduce to
one condition and leave the input bit intact. The append test checks the
full phase and amplitude for both classical input values. Both source and
native refuse duplicate quantum wires. The source checks on circuit
finalization and native validates at instruction construction.

There is no allocated workspace or measurement output. The
duplicate-bit fixture emits one conditional NEG in KMX. The declared
cost is the emitted stream, with no independent analytical model.
Source scope parsing and typed IDs remain in their own modules.

Run `python -m pytest -q tests/unit/test_kickmix_cz_binding_file.py`.
