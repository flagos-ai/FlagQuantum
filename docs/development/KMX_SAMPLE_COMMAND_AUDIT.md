# Physical register sampling

K-0040 and K-0041 track the locked `main_sample.cc` and its original test
(blobs `53999a8306614b44e833093ecc526080e8c2903c` and
`fe810b71ffb5c07772db1c90d2a306bc4a803d3c`).

Run `python -m flagquantum.ecosystem.kmx_sample --in circuit.kmb --shots 65`.
Omitting the file options uses stdin and stdout. The command shares the
physical readers and the sampled-count command's initialization parser.
Simulation snapshots initialized packed wires on request and reports
both the exact accumulated phase and the separate sign bit. Formatting
uses declaration order, least-significant-first member lanes, the locked
before/after header and padding, and hex precedence over decimal.

The source's CSV `global_phase` field reads the sign bit. A `Z_POW`
rotation contributes to the exact phase without changing that bit.
An operand-free debug record reads the angle accumulator while the
current scan's sign remains in local state. Quantum and classical debug
operands print shot zero if its guard is retained. The command emits
these events to stderr once per physical batch, including a partial
batch, and produces no debug events for zero shots.

## Verification

The original named/unnamed register test passes against unchanged
source. Eleven deterministic locked reports match native bytes for both
KMX and KMB. The fixtures exercise a 70-bit register, an empty register,
hex/decimal precedence, fixed mixed initialization, the sign and angle
split, debug guards, and zero and 65-shot runs. The pipe path separately
matches both output streams. Invalid requests preserve an existing
output file.

For the random-initialization and HMR case, the source and native
65-shot tables cover all eight combinations of initial word and HMR
outcome. Each row preserves input ordering, conditional X feedback,
the exact scalar sign, and the cleared work bit. The shared simulator's
earlier locked-history regression checks the corresponding full complex
amplitudes through the append workflow. A separate two-shot tape checks
signed phase and its rotation component in exact fixed-point units.
The native command owns its random stream, so randomized rows compare
by history. Deterministic reports compare byte for byte.

Sampling executes the existing stream without emitting extra quantum
operations. The reported columns are observations, with no resource
estimator involved.
