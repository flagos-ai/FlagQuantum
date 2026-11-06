# Direct quantum variable rotation

The shared emitter ports the concrete QvrZPow schedule from locked
Qualtran v0.7.0. The big-endian cost register carries an explicit
fractional width. The source retains the lower of that width and
`ceil(log2(2π |coefficient| / epsilon))` fractional positions, then
visits retained wires from least to most significant. Each term
records its wire, Z exponent and source synthesis tolerance. The
native IR emits the corresponding phase angle in radians. The
adjoint visits the same terms in reverse with opposite angles.

The original graph and native gate stream agree on the tested
unsigned inputs. The tests also cover the exact complex phase on
every basis word and a coherent spectator, including reversal.
The emitted phases use no work wires or measurements.

Several source precision boundaries are visible in the literal
graph. A zero coefficient makes the logarithm undefined, and a zero
rotation count makes the tolerance division undefined. The native
planner reports both explicitly. A negative rotation count skips
the loop. For unsigned input it emits no operations, while the
original model can report a negative rotation estimate. For signed
input it retains the source's standalone Z on the sign wire. This
is literal graph behavior. The signed source with a positive
rotation count indexes past the register, so that layout is
rejected explicitly.

| Cost width, fractional width, coefficient, epsilon | Native phases | Native T-family | Work and measurements | Source model |
| --- | ---: | ---: | ---: | --- |
| 4, 4, 0.1, 0.5 | 1 | 0 | 0 | 1 rotation |
| 5, 2, -0.375, 0.05 | 5 | 0 | 0 | 5 rotations |
| 5, 0, 0.125, 0.1 | 5 | 0 | 0 | 5 rotations |
| 8, 8, 0.625, 0.03 | 8 | 0 | 0 | 8 rotations |
| 4, 4, 0.0001, 0.5 | 0 | 0 | 0 | -9 rotations |

The native T-family column counts emitted IR before synthesis.
The source model repeats a representative rotation. For the five-bit
integer case at coefficient 0.125, the exact source leaves instead
classify as one rotation, one T and three Cliffords.

Run `python -m examples.algorithms.register_qvr_zpow` to check
the direct phase through `circuit.append(op)`.
