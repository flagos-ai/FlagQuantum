# Uniform preparation over a concrete range

`uniform_preparation_ir` binds the first N basis states of a big-endian
register of width ceil(log2(N)). Controls are ordered explicitly and
may have mixed polarity. The forward entry takes a clean target when
used for preparation. The adjoint reverses the complete circuit.
Both directions preserve control wires and return all borrowed quantum
work to zero. Classical slots may be overwritten.

## Pinned circuit and phase

The [locked source](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/prepare_uniform_superposition.py)
factors N into an odd part L and a power of two. It applies the initial
Hadamards under the supplied controls. For L greater than one, the
remaining stages act only on the high bits. In source order they apply
a phase conditioned on comparison against L, Hadamards, a phase
conditioned on the high bits being zero and the controls matching, and
final Hadamards. Both conditional phases use Rz on a predicate qubit.
Replacing Rz by a one-sided phase would change the complex amplitudes.

For example, with N=3 the amplitudes on 0, 1 and 2 are
`1/3 + i*sqrt(2)/3` for the matching control pattern. A nonmatching
pattern leaves a clean target unchanged. The test also compares the
source's action on arbitrary targets, where the inactive branch can
carry a phase. Comparisons preserve both the relative phase across
coherent controls and the source's overall phase convention.

The threshold stage reuses the literal LessThanConstant and its
self-inverse circuit. The zero predicate reuses the source And/MultiAnd
ladder and measured erase. The source yields each controlled Hadamard
separately. The native path retains that boundary. Each multi-controlled
Hadamard computes its own temporary flag, applies CH, then erases the
flag. The same work is available again for the next target. The existing
Clifford+T translation requires two T-family gates per CH.

Power-of-two counts use just the initial Hadamard stage. The singleton
is identity on an empty target. The locked generator emits nothing
when supplied an explicit empty target, but its generic Cirq adapter
raises a missing-target error for this case.

## Resource evidence

These are per direction. The native and source streams have identical
instruction order, operands, measurement feedback and opcode counts
after both streams use the same lowering of each controlled Hadamard
to the measured And ladder and Clifford+T CH. Rz remains an explicit
rotation on both sides. The last column separately records Qualtran's
concrete Cirq T-complexity model, which uses its own rotation accounting.

| Count | Control pattern | Work qubits | Classical slots | Native CH | Native T + T† with CH lowered | Measurements | Native Rz | Qualtran model T / rotations |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 3 | None | 4 | 2 | 0 | 20 | 5 | 2 | 20 / 2 |
| 3 | 0 | 4 | 2 | 2 | 28 | 6 | 2 | 24 / 6 |
| 5 | 1,0,1 | 5 | 5 | 3 | 74 | 17 | 2 | 68 / 8 |
| 6 | 1,0 | 4 | 3 | 3 | 46 | 10 | 2 | 40 / 8 |

The tests bind the unchanged source generator, compare the comparison
and reflection leaves including feedback, and compare exact complex
vectors from the source forward and adjoint with occupied targets and
a spectator. Through the append entry point they exhaust the raw
measurement histories of the smallest circuits, sample larger mixed
control circuits, and check forward/adjoint restoration and clean work.


## QEC model profiles (G2-PREP07)

`source_uniform_preparation_cost(count, legacy_shims=True)` retains the pinned
legacy QEC classification. It queries the same shared And authority as the
modern default, including the negative zero predicate and measured cleanup.
`source_equal_width_less_equal_cost(width, legacy_shims=True)` likewise charges
the source comparator tree's positive And pairs. The original comparator call
graph declares those pairs independently of polarities in its gate generator.
Neither function emits instructions, changes the circuit or substitutes a
primitive Clifford count for the source resource categories. Both directions
have the same declared costs, and peak logical width is unchanged by profile.

Fourteen comparator and 24 uniform configurations compare all source QEC
categories and peak widths in both directions. Invalid nonboolean profile
flags are rejected. Existing source gate-stream, complex-state, branch-history
and clean-work regressions cover the unchanged emitters.

This dependency batch has 42 new tests and 180 concentrated passing tests
without skips; strict types, formatting, lint and architecture checks pass.
