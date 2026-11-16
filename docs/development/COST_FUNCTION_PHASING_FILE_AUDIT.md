# Cost-function phasing source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `rotations/__init__.py` | `e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8` | Q-0666 |
| `phasing_via_cost_function.ipynb` | `b1c45ff7e36c3d82b43166e98c39d5ba89c68762` | Q-0673 |
| `phasing_via_cost_function.py` | `7e4204a0dcc72fa5785b2e58f4c209d0f083dd76` | Q-0674 |
| `phasing_via_cost_function_test.py` | `959d1c2e1b94b83c505aa8d27747002c61540406` | Q-0675 |

## Tutorial and port bindings

Run `jupyter notebook examples/notebooks/phasing_via_cost_function.ipynb`.
The native notebook preserves the twelve original cells and adds a body
compilation example using `circuit.append(op)`. The two original factories
use five input bits, ten fractional cost bits, coefficient 0.1234, and
tolerance 1e-8. Inputs and the caller-owned 32-bit gradient are through
registers. The ten-bit internal result and scratch wires are initialized to
zero and restored after the phase body. The source order is
`Square → QvrZPow / QvrPhaseGradient → Square†`.

The native notebook rendered three SVGs in an independent kernel with
Qualtran, Cirq, and quimb imports forbidden. The original notebook also ran
in an independent kernel.

## Source grids and measurement evidence

The original suite returned 10 passes, two skips, and 29 expected failures.
The Hamming grid explicitly marks 24 Cirq simulation cases as expected
failures. Five Square cases hit the source flattening boundary. The source
autotester skipped the inferred call graph and the `QUInt(10) → QFxp(10,10)`
type check.

The native grid checks both register encodings for each of the 24 original
Hamming and 12 original Square parameter combinations. The original
cost and rotation graphs determine the phase and gradient translation.
For comparable leaves the tests compare gate names, operands, angles,
measurement slots, and feedback. The source Square leaf has no physical
decomposition. The comparison binds its published classical permutation
to the independently verified column compressor. The gradient walker
follows the original scaled-add Cirq children to their bound Add leaves.

The measurement tests project every small raw history. For larger circuits,
the balance and clean-work certificate checks every erasure invariant over
all input values with either all small gradient words or representative
large words including both ends of the word range. Two sparse trajectories
compare the full supplied complex states. The 32-bit tutorial body is
verified over all five-bit inputs and six gradient basis words, including
wrap boundaries. Full Fourier-state kickback execution is checked on the
smaller fixtures.

| Five-bit body | Literal T/T† | Literal CX | Literal phase | Literal measurements | Source QEC categories | Reserved / source wires |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Square with Z powers | 160 | 394 | 10 | 40 | 40 Toffoli, 10 rotations | 35 / 15 |
| Square with gradient | 1636 | 4062 | 0 | 409 | 520 Toffoli | 98 / 47 |

Measurements and width describe the compiled body. The caller owns gradient
preparation and optional unpreparation. The direct phase gates are counted
before synthesis.

## Remaining source boundaries

Q-0673 and Q-0675 are covered. Q-0674 stays open for generic symbolic and
external oracles and the original Square gate decomposition and mixed
fixed-point type boundary. The current composer accepts already bound
native emitters with explicit cost output names and wire ownership.
Q-0666 stays open for the remaining rotation exports.
