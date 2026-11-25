# Programmed ancilla rotation source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `rotations/__init__.py` | `e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8` | Q-0666 |
| `programmable_ancilla_rotation.py` | `89561f7a56e1567210a14ee6af90ee65505c3172` | Q-0676 |
| `programmable_ancilla_rotation_test.py` | `7882ccc1522dc4abe2568e1e2d636d31f837faea` | Q-0677 |

Run `python -m examples.algorithms.programmed_phase_catalog` to append the
concrete source fixture. Preparation binds the original Allocate, H, and Z
power to a clean caller-supplied output. The operation has a single RIGHT
qubit. Programmed rotation has a single THRU data qubit. Native execution
reuses one work qubit and records one classical slot per round.

## Source declarations and original tests

Both preparation fixtures and all three programmed rotation fixtures were
compared against their original signatures and direct call identities,
including the exact symbolic angle and precision expressions, physical
synthesis tolerances, and rotation multiplicities. The preparation tests
map the original five random angles from seed 42 against source tensors.
The resource tests compare concrete QEC categories, symbolic widths,
the failure-bound expression, and the precision-sensitive T estimator.

For `n` symbolic rounds, the source declares `3n` Clifford gates, `n`
measurements, and `n` rotations, including H inside each ancilla preparation.
The source chooses `ceil(log2(1/p))` rounds for maximum failure probability
`p`. The rotation precision is divided by the number of declared ancillas
and the optional final correction. The `p=1` fixture binds an identity
with zero rounds. A zero-round corrected operation emits one unconditional
phase. The native lower-level helper continues to require nonempty slots
for its measured rotation body.

The original suite reported eight passes and five skips. Skips cover
decomposition and dtype checks for the adaptive Bloq, the source split
between decomposition-derived and declared counts, and measurement
serialization.

## Literal execution and source model

| Concrete default fixture | Literal H | Literal phases | Literal CX | Literal measurements | Literal X | Source logical categories |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Preparation, one output wire | 1 | 1 | 0 | 0 | 0 | 1 Clifford, 1 rotation |
| Rotation, two reserved wires | 2 | 2 | 2 | 2 | 2 conditional | 6 Clifford, 2 rotations, 2 measurements |

The source estimates 106 T gates for synthesis of the default two-round
rotation at the specified tolerance. Its literal circuit contains the
angle-valued phases before synthesis. The optional last correction adds
one conditional phase after the final failure.

All raw histories for up to three rounds were projected on a complex
entangled data and spectator state, including the zero vector for impossible
histories. The comparison includes branch amplitudes and global phases,
conditional work reset, and the corrected and uncorrected terminal outcomes.

## Remaining source boundaries

Q-0677 is covered. The original
adaptive Bloq supplies a worst-case call graph and omits a gate
decomposition. In the native flat circuit, inactive rounds measure a clean
zero work qubit as padding in the original compatibility schedule. Symbolic declarations are modeled before
binding concrete execution parameters. Q-0666 remains open for the other
rotation exports.


## Physical early-exit closure, 2026-11-26

Q-0676's remaining execution requirement is now implemented through the existing
internal guarded-effect contract. The normal example append path requests early
exit; later measurement destinations receive a classical zero initializer, then
the existing shared guard owner predicates each measurement on all preceding
failures. A successful attempt therefore executes no later measurement, while
fixed-width result records keep zero suffixes. The source remains model-only
for consumption; this is the separately specified Jones channel, not an invented
Qualtran primitive body.

Full unnormalized complex amplitudes agree with the certified padded route for
all histories through four rounds, including impossible histories, arbitrary
slot order, spectators and exact failure scalars. Both local execution strategies
verify the number of actual measurements against attempted rounds. Density
channels agree with and without final correction. The work returns to zero.
Prepared output ownership, all five source factories and resource-model evidence
are retained. Composed symbolic queries now use the same shared fresh-angle
allocator as the original source, avoiding accidental node coalescing.

Three rounds still emit three potential H, phase, CX, measurement and conditional
X instructions, plus two classical initializers; only active quantum effects
execute. There is one reusable work qubit and three caller-owned outcome slots.
For generic exponents the separate original worst-case model remains nine
Cliffords, three measurements and three precision-sensitive rotations. No primitive-to-model equality or
unimplemented source consumption body is claimed. Public conditional-measurement
construction and unsupported typed result merges remain unchanged.


The concrete exponent 1/8, three-round, 256-shot receipt executes 444
measurements in trajectory mode and 448 in batched mode (seed 42), versus
768 padded measurements. This is a measured per-run count, not a worst-case
estimate. The unchanged source model classifies that special angle as ten
Cliffords, one T, one rotation and three measurements; terminal correction adds
one Clifford. Both synthesis estimates are 54 T. Different strategies need not
consume the same random stream, and each is checked against its actual histories.
