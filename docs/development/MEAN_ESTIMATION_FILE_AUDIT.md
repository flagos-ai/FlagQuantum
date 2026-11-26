# Mean estimation source audit

The reference is unchanged Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Asset | Git blob | Ledger item |
| --- | --- | --- |
| `mean_estimation/__init__.py` | `ce76a8d10a5909daeb7b1113c8cc7f26cc31b410` | Q-0563 |
| `mean_estimation_operator.py` | `1e5b5d7db7b2c397fad8107d3c0e330958c2bb7e` | Q-0566 |
| `mean_estimation_operator_test.py` | `bbc6d9ff32f223237e9506892cb5cb9ed53176fe` | Q-0567 |

The source package initializer contains only its license header. The two
native operations live under the existing Algorithms primitives. No package
export is needed for the empty source surface.

## Binding and order

`CodeForRandomVariable` requires the synthesizer and encoder selection
registers to agree. The caller supplies both bound coherent circuits to
`MeanEstimationOperator.make_on`, then uses `circuit.append(op)`. Controls
and selection wires are live through ports. Encoded values, lookup data
and reflection work use one caller-owned clean pool and are returned to zero.
The phase step finishes cleanup before the reflection borrows that pool.
The native inverse runs the reflection first and then applies the inverse
phase. The locked `-1` in the reflection is emitted as a π phase on its set
temporary flag. The zero mark, feedback, temporary flag roles and primitive
operand order are compared with the locked source.

A single outer control uses either polarity. The native phase step computes
and erases the encoder and lookup unconditionally and controls only the phase
between them. One reusable And scratch wire and one slot combine the outer
control and the sign bit. The reflection is controlled through the shared
reflection emitter. For a power, each step reuses the same clean pool. The
logical call graph retains the original phase and reflection children,
positive-control wrappers, power multiplicity and the two source X calls
on the zero-control wrapper. Native physical zero controls are lowered
inside the two stages.

## Verification and resources

The unchanged source suite passes all 12 tests. The native audit checks
full unitary matrices for the eight original Bernoulli and three Grover
fixtures, and applies the original spectral-overlap criterion to the
native matrices. The Grover encoder uses the existing exact ancilla-free
Circuit control synthesis for wider opaque multi-controlled X calls.
For the two-bit Bernoulli witness, every one of the 32 measurement
histories is checked in each direction with a full complex input and zero
workspace output. The single-bit controlled witness checks both polarities,
positive and inverse powers, and every measurement history. It matches
the direct controlled source matrix and its labels and register order.

| Native fixture | Native wires | Emitted T/T† | Emitted CX | Emitted CCX | Emitted controlled phases | Emitted measurements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bernoulli, two selection bits, two target bits, precision five | 11 | 20 | 56 | 4 | 10 | 5 |
| Bernoulli, one selection bit, one target bit, precision two, zero-controlled square | 7 | 32 | 60 | 0 | 8 | 8 |

For the first row the uncontrolled source model has peak width 22,
`400*_c0 + 1` T, four Toffoli, nine And, 25 Clifford, ten rotations and nine
measurements. Its opaque ArcTan declaration and controlled-power
denominations differ from native emission. The implemented source resource
method prices one forward uncontrolled step. Source zero power refuses
construction, and the native implementation also rejects zero power.

Q-0563 and Q-0567 are accepted. Q-0566 remains open for source resource estimates of controlled or powered
forms and the retained accuracy/ABI boundaries. The live typed context now
closes concrete multiple external controls through the actual positive source
child and shared predicate owner; see the dependency audit below.
The existing flat operation uses exact bounded integer encoding and coherent
children on the declared ports. The additional
[typed dependency context](MEAN_ESTIMATION_COMPONENT_AUDIT.md) owns explicit
encoder/PREPARE pairs, including measured encoder work and demonstrated clean
synthesizer junk. Explicit caller-owned signal junk preserves the full source
physical action when private cleanup is not valid; this is an expanded native
signature, not the smaller source ABI. The same audit retains a default-32
source SDK phase-decomposition accuracy failure. None of these context checks
closes the outstanding parent scope or fabricates source code serialization.
