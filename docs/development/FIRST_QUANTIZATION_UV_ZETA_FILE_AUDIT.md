# First-quantization potential and nuclear-index PREPARE audit

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/pbc/first_quantization`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0370 | `prepare_uv.py` | `2c5e79f0c5b743a444d63e0b00048cd50c8c4060` |
| Q-0371 | `prepare_uv_test.py` | `266e2557bf7b629eb24b0ea44f8744adfabf859a` |
| Q-0372 | `prepare_zeta.py` | `dc76e1ccf9b86d3107da6aad37daed120d0981d1` |
| Q-0373 | `prepare_zeta_test.py` | `55cb3afc0bd25469f7b2d0e70c8e869522bd1b62` |

## Register and child contracts

Potential PREPARE declares `mu, nu, m, l, flag_nu` in that order.
The ν child owns `mu, nu, m, flag_nu`, including the three separate
momentum axes. The nuclear-index child owns `l`. Both directions
preserve each named word and indexed element at its own parent port.
At one atom the nuclear index remains an opaque zero-width word.
The model and tutorial share these signatures, and the resource
comparison reads the live bystanders from the original two-call graph.

The ζ source keeps nuclear-coordinate precision as metadata. Its
resources depend on the nuclear charge and direction. The forward
formula charges `lambda_zeta`; the adjoint charges the ceiling of the
floating-point square root. The native model preserves the source's
floating-point rounding, including the large-integer boundary.

## Execution and resources

The original UV example composes a ν child and a ζ child. The ζ child
has no gate decomposition. The existing ν evidence covers its runnable
m substep and missing physical leaves. The native UV binding tests
append the m substep on the parent m wires with the other through words
present, and confirm that the gate stream touches only m and its work
wires. The same three-value substep has independent full-amplitude and
cleanup comparisons for all 32 histories in each direction.

At p=6, M=32768 and total charge 10, the model estimates 2,516 T
equivalents across forward and adjoint, with peaks 77 and 50 logical
qubits respectively. The independent three-value m fixture emits
20 T/T† and five measurements with four work qubits. These figures
describe separate scopes.

## Original tests and serialization

The unchanged source suite reported eight passes and three skips in the
isolated Protobuf 5.29.5 reference environment. Both serialization
checks ran and passed there. All three skips are ζ checks that require a
decomposition, including the composite-count and dtype checks.
The earlier Protobuf 7.36.2 run rejected the legacy bool argument written
to an integer protobuf field. That runtime compatibility issue remains
recorded. The native register code does not depend on that serializer.

Q-0370 through Q-0373 close the two source files and their tests within
the frozen reference scope. End-to-end potential PREPARE still needs
physical implementations of the undecomposed ν and ζ leaves.
