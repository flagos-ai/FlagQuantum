# Projectile kinetic and potential PREPARE audit

The subsequent [finite potential constructor audit](RESOLVER_POTENTIAL_PREPARATION_AUDIT.md)
connects both original UV constructors and the zeta declaration to the shared
typed nu dependency. It preserves the source inverse/physical boundaries below;
constructor registration does not close those gaps.

The [finite kinetic constructor audit](RESOLVER_KINETIC_PREPARATION_AUDIT.md)
adds all four ordinary/projectile T and unary constructor bindings. Only the
ordinary T has the original uniform/r/s physical call graph; all opaque source
leaves and negative narrow declaration counts remain explicit.

The frozen Qualtran v0.7.0 reference is under
`qualtran/bloqs/chemistry/pbc/first_quantization/projectile` at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0377 | `prepare_t.py` | `211c6cc15d39f2612c8be6ad1040612b7c2e7cc2` |
| Q-0378 | `prepare_t_test.py` | `a250da50e0dfea6a3a8740da7a43da9081865231` |
| Q-0379 | `prepare_uv.py` | `f66875a9c0855bf1dab802d8ea7064bb80d017ad` |
| Q-0380 | `prepare_uv_test.py` | `b516f4c5c3eadf2c176327bd0e9320f82473e423` |

## Kinetic register contract

The native T declaration has separate two-bit through words `w` and
`w_mean`. The `r` and `s` words use the projectile momentum width.
The power-of-two leaf has one through word of that width. Both
directions reuse the ordinary kinetic register owner and the source
charge for the projectile-specific coordinate tests. At electron width
6 and projectile width 8 the source budgets 47 forward and 27 adjoint
Toffoli, with a peak of 20 logical qubits in each direction. The
source provides no T composite or power-of-two leaf decomposition.
The shared ordinary coordinate preparation remains independently
executable through `circuit.append(op)`.

## Potential-energy composition

UV reuses the common through signature with projectile-width momentum.
The ν child owns `mu, nu, m, flag_nu`; ζ owns only `l`. A one-atom
index retains its explicit zero-width port. The frozen outer adjoint
can reverse the two calls, retaining the same parent inputs and outputs.
Its nested ν decomposition and recursive peak query still fail
because the source μ adjoint leaves its temporary flag on the right.

The corrected native UV estimate composes the native ν adjoint model
with the ζ adjoint. Live bystanders follow the original outer bindings.
For electron width 6, projectile width 8, 10 atoms, M=524288, and
total charge 10, the forward estimate is 3,908 T equivalents with
a peak of 97 logical qubits. The original adjoint declares 132 T
equivalents; the corrected native model uses those gate categories
and estimates a peak of 62 logical qubits. Expanding its ν child
also accounts for 19 Clifford gates for uniform m at that size.

The separately executable three-value m fixture emits 20 T/T† and
five measurements per direction, with four work qubits. The preceding
batch compared its full complex amplitudes and zero work for all 32
histories in both directions. No full T or UV emitted count is available.

The untouched original suite reported 12 passed and six skipped.
Q-0377, Q-0378 and Q-0380 close the kinetic declarations and original
test coverage. Q-0379 stays open for the locked recursive adjoint
failure and missing end-to-end physical preparation leaves.

## Complete logical momentum and potential dependency closure (2026-11-28)

The [complete potential constructor closure](RESOLVER_POTENTIAL_PREPARATION_AUDIT.md#complete-logical-momentum-and-potential-dependency-closure-2026-11-28)
adds zero/symbolic declarations, shallow graphs with invalid recursive Nu,
exact outer inverse and independent symbolic categories/peak queries.
Q-0379 is eligible for faithful-source closure after checks and publication.
This does not close the separate T-file or top-level consumers. Source opaque
physical leaves, metadata codec loss and numerical inverse-rounding boundaries
remain recorded.
