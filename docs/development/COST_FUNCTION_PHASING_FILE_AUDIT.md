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

Q-0673 and Q-0675 are covered. Q-0666's export inventory is accepted.
Q-0674 remains open for the unchanged source Square decomposition and mixed
fixed-point type boundary. Concrete binding of symbolic fields and external
RIGHT cost oracles is now exercised; unresolved symbolic values are rejected
before physical execution. This does not promise execution for every external
oracle or every fixed-point format.

## Complete source binding and ownership

The whole source file maps as follows:

| Source definition | Owned implementation and evidence |
| --- | --- |
| `cost_eval_oracle`, `phase_oracle` and post-init validation | Three bound stages in `compiler/registers/phase/cost_function_phasing.py`; source-side RIGHT cost and THRU phase admission is retained. |
| Cached `signature` | Source left ports and phase extras bind to named native registers. Cost output is private; the gradient remains caller-owned. |
| `build_composite_bloq`, including `_extract_soqs` | Compute, phase, and true oracle adjoint at the same `BlackBoxPrepare` append consumer, with named output and scratch ownership. |
| Both Square factories | Actual original objects are imported in both directions; smaller resolved instances exercise complete logical operators and every erasure history. |
| Documentation specification | Both original examples remain in the native notebook and catalogue; earlier independent kernel evidence is retained. |

`test_cost_phasing_source_binding.py` supplies a real optional-source oracle
with two copy CNOTs into a newly allocated `QFxp(2,2)` cost register. Symbolic
fields are explicitly resolved, then both signs of the coefficient and both
QVR strategies are captured without changing the reference. Native callbacks
replace these three stages at the same append consumer. Every logical column,
including every small gradient basis word, agrees in full complex amplitude;
all private cost and carry wires return to zero. The tests check owned JSON
round trips, true separately captured adjoints, all-history channel identities,
and exact opcode histograms. Finite-precision Fourier kickback is tested against
its approximation tolerance, separately from exact basis-word translation.

The optional Cirq boundary now recognizes the exact source `Square` leaf and
delegates to the existing native integer-square column compressor. This is a
realization of the source's value/isometry contract using the paper construction
(arXiv:2105.12767, page 76), **not** a newly discovered source gate recipe.
The frozen arithmetic source blob is
`efed2d6652c7dcc943c82e0c5d35ae5b3b627c09`. Its decomposition still raises
`DecomposeNotImplementedError`; no reference is patched. Widths one through
three compare every valid input against the original numerical tensor and
restore input/result/work with the true source adjoint. RIGHT result starts
at zero; LEFT consumes the matching square. Arbitrary invalid LEFT inputs
are outside that source contract. Reusable measurement storage and carry
allocation remain owned by the existing exporter and qubit manager.

The original five-bit direct factory is also executed on a coherent state
containing all 32 input values with two sparse trajectories. Its every-history
erasure certificate is checked separately. The original 32-bit-gradient factory
is captured in both directions and its primitive counts are checked; this is
not a claim of exhaustive execution over all 37 logical input bits. Earlier
large-word arithmetic evidence and the bounded full operators remain separate.
The table above describes the native tutorial's reserved layout; imported
source snapshots can reuse lifetimes and reserve fewer physical wires.

## Validation of the binding extension

The two new test files pass all 12 cases (8.46 seconds); 668 concentrated
regression cases pass (83.70 seconds), followed by the final focused run after
adding the original gradient-factory capture. CPU/representation integration
passes 60 cases (41.29 seconds), and the native cost-function notebook runs
in an independent kernel (2.49 seconds). Strict typing of all four changed
Python files, Black, Ruff, Ecosystem scope, architecture, multilevel IR, and
protected API checks pass. Serial branch coverage of the two adapter files
is 82% combined; the new square adapter is 88%. Defensive admission guards
remain uncovered, so this is not a complete branch-coverage claim.
