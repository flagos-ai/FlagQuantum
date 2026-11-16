# Public resource generalizers source-file audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Source behavior | FlagQuantum owner |
| --- | --- |
| `ignore_split_join` | `ignore_register_partition` drops split, join, partition and cast keys and unwraps nested partitions. |
| `ignore_alloc_free`, `_ignore_wrapper` | `ignore_allocation` and `_inside_partition` omit allocation and release and retain layouts on surviving wrapped keys. |
| `generalize_rotation_angle`, `PHI` | `generalize_rotation` uses an unspecified angle or exponent in the native key, and groups S/T adjoints with their forward keys. |
| `generalize_cvs`, `CV` | `generalize_control_values` uses unspecified binary controls and an arity-only key for forward MultiAnd. |
| `generalize_cswap_approx` | `generalize_approximate_swap` maps both directions to the exact self-adjoint swap resource identity. |
| `ignore_cliffords` | `ignore_known_cliffords` follows the native classification predicate and preserves layouts on surviving wrappers. |
| `cirq_to_bloqs` | `generalize_cirq_resource_call` recognizes SDK keys within Ecosystem through `cirq_gate_resource_key`. Known keys retain the ordered layouts. Unsupported gates stay opaque at the adapter boundary. |

Q-0814 covers all eight unmodified original tests. The native tests compare
the locked 16-entry ordered fixture, including direct and wrapped CNOT/S,
forward and adjoint approximate swap, nested rotations, and omission.
Independent tests compare weighted graph folding and admitted SDK gate costs.
An SDK Clifford remains opaque before conversion. Converting first makes it
eligible for removal by the native Clifford rule.

Q-0813 admits an exact inverse once its SDK decomposition resolves to
owned resource keys. The ordered native sequence keeps adjoint T distinct
from forward T and remains within the existing call-count and QEC query
paths. A one-clause Cirq predicate converts to a native control pattern.
Multiple clauses raise a descriptive error, matching locked source admission.
Unrecognized SDK gates and incomplete decompositions retain opaque adapter
identity. General SDK wrapper metadata and ancilla lifetime import remain
tracked by the dedicated Cirq conversion source files.

The model rewrites emit zero primitive gates. The separate inverse import
binds CNOT, T-adjoint and H in source order, preserving wire operands and
complex amplitudes through normal append. Self-inverse gates with exponent
minus one and inverse S/T phases import losslessly at zero global shift.
