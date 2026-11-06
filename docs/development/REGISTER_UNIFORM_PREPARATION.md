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
ladder and measured erase. With multiple outer controls, one additional
conjunction drives the initial CH gates and is then erased. The source
uses Cirq controlled Hadamards directly. FlagQuantum preserves their
unitary and uses a native CH with a shared flag. Its existing
Clifford+T translation requires two T-family gates per CH.

Power-of-two counts use just the initial Hadamard stage. The singleton
is identity on an empty target. The locked generator emits nothing
when supplied an explicit empty target, but its generic Cirq adapter
raises a missing-target error for this case.

## Resource evidence

These are per direction. The emitted T-family column includes the
existing exact CH translation. Rz synthesis is separate. The source
columns are Qualtran's concrete Cirq T-complexity model.

| Count | Control pattern | Work qubits | Classical slots | Native CH | Native T + T† with CH lowered | Measurements | Native Rz | Qualtran model T / rotations |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 3 | None | 4 | 2 | 0 | 20 | 5 | 2 | 20 / 2 |
| 3 | 0 | 4 | 2 | 2 | 28 | 6 | 2 | 24 / 6 |
| 5 | 1,0,1 | 5 | 5 | 3 | 58 | 13 | 2 | 68 / 8 |
| 6 | 1,0 | 4 | 3 | 3 | 38 | 8 | 2 | 40 / 8 |

The tests bind the unchanged source generator, compare the comparison
and reflection leaves including feedback, and compare exact complex
vectors from the source forward and adjoint with occupied targets and
a spectator. Through the append entry point they exhaust the raw
measurement histories of the smallest circuits, sample larger mixed
control circuits, and check forward/adjoint restoration and clean work.
