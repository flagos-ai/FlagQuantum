# Alias probability preprocessing source audit

Algorithms owns classical coefficient preprocessing for its existing coherent
alias PREPARE consumer. This is the `single_device_fast_path`; the helper neither
executes quantum circuits nor chooses a backend. The reference is unchanged
Qualtran v0.7.0 `linalg/lcu_util.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Native code retains the source copyright
and Apache-2.0 identifier. Six source Git blobs are checked against the frozen
inventory in workspace `g2-alias-discretization-source-audit-source.json`.

## Whole source disposition

| Original definition | Native target in `algorithms/primitives/alias_discretization.py` |
| --- | --- |
| `_partial_sums` | Private generator; initial integer zero, source consumption/yield order |
| `_differences` | Private adjacent-difference generator; empty and singleton behavior |
| `_discretize_probability_distribution` | `discretize_probabilities`: ordered sums, cumulative floor rounding, then differences |
| `_preprocess_for_efficient_roulette_selection` | `integer_alias_rows`: input copy, divisibility checks, two passes with monotone donor position |
| `sub_bit_prec_from_epsilon`, both overloads | `alias_precision_bits`: existing Core log2/ceil/symbolic classifier, concrete minimum one bit |
| `preprocess_probabilities_for_reversible_sampling` | `reversible_sampling_rows`: exactly one precision choice, original failure order, mutable list result |
| All three original unittest classes, their helper methods, and standalone zero-width test | All 11 tests run unchanged; all 11 run with only four helper bindings replaced |

No other functions, classes, exported constants or assets occur in the
implementation file. The private generators are retained because rounding uses
them; they are not additional public concepts. Existing Core scalar arithmetic
and symbolic classification remain authoritative. No dependency, registry,
public export, protected API or alternate IR is introduced.

The original `linalg/__init__.py` contains only a package docstring and license;
`linalg/polynomial/__init__.py` contains only its license. AST and original Git
blobs confirm no exports, imports or executable statements to port. Their
namespace-only roles are disposed into existing Algorithms, Compiler and
Simulation owners, without creating original-package mirrors. This closes
Q-0754 and Q-0761 as two explicitly audited namespace assets, not new algorithms.
Apache attribution remains on the adapted implementations and existing repository
notices; no source license is removed.

## Boundary, interfaces and replacement

Raw helpers reproduce the reference arithmetic and errors. They do not impose
the stricter finite/nonnegative domain of `AliasSamplingTable`. Empty, zero-sum,
negative, NaN/infinite and unsuitable precision inputs retain observed source
values, exception type/message and warning order. The annotated partial-sum
accumulator starts with the literal integer zero, without a float conversion;
large integer tests check that behavior. Input rows are not mutated.

Validated `alias_numerators` retains its existing checks, then calls the shared
raw preprocessor. Dense/sparse table factories retain absolute/normalized
precision respectively. Their width calculation and the validated symbolic
shape entry now share `alias_precision_bits`, removing duplicate formulas and
the old cumulative/donor implementation. Native validation exceptions remain
native; no claim equates them to the raw upstream domain.

The ordinary workflow is unchanged: construct a validated table and append the
existing PREPARE operation, as in
[the register guide](REGISTER_ALIAS_PREPARATION.md) and
`examples/algorithms/linear_combination_controls.py`. Logical registers remain
selection, sigma, alternate, keep and comparison flag, plus sparse index where
applicable. Register words are big-endian; marker/carry work and classical slots
remain caller-owned. Logical PREPARE junk is not disposable work. The adjoint
clears matching prepared logical registers; temporary work clears in both
execution directions.

Source/native preprocessing is also exercised with replaced width, rounding and
donor providers, retaining call order and arguments without changing consumers.
Existing original source PREPARE stage/gate comparisons continue through the
validated factories after their implementation replacement. There is one
rounding and donor owner, with no compatibility registry or parallel sampler.

## Local verification

- 144 new tests pass, including exhaustive weights 0..3 for lengths 1..4 and
  widths 1..4, all small raw integer donors including negative weights,
  concrete/symbolic widths, optional-argument failure order, warning semantics,
  lazy generators and large-integer sums.
- All 11 original tests pass unchanged. All 10 unittest methods plus the
  standalone test pass again with native helper bindings; the tests' random
  state is restored afterward.
- Concentrated regression: 278 passes, no skips; six existing reference Autoray
  numerical warnings are recorded separately.
- New owner coverage: 65 statements and 26 branches, 100%. Strict typing covers
  three production files and the new audit test. Two reused consumer helpers
  receive precise annotations; their wider pre-existing test file is not
  represented as strict-typing certified.
- Black/Ruff cover all five changed Python files. Algorithms scope, architecture,
  internal IR and API baseline checks pass. Default-suite results and baseline
  failure comparison are recorded in the live ledger: 7 failed, 21,651 passed,
  300 skipped and 1,037 warnings in 190.83 seconds, stopping at the existing
  maxfail bound. The seven failure IDs exactly match the previous baseline;
  no new failure is observed. This is not a full-green default suite.

Three two-row consumers use weights `(1,3)`, `(3,1)` and `(1,1)` at one probability
bit. Native/raw source tables produce identical emitted instructions. Forward
and inverse execution check every measurement history (12 branch evaluations),
full complex amplitudes including unnormalized branch factors, and all temporary
work amplitudes within 3e-12 of zero. Existing dense/sparse source stage and
primitive-stream comparisons run in the same regression. Neither arbitrary
large measurement histories nor approximate global-phase equality is claimed.

## Measured primitives versus model

Preprocessing alone emits zero quantum gates, consumes zero quantum work/feedback
slots, and has no source QEC model. It uses linear table/cumulative storage and
source-ordered classical arithmetic. No hardware performance claim is made.

| Weights | Measured forward primitives | Source/native model |
| --- | --- | --- |
| `(1,3)` | H7, X9, CX26, T6, Tdg5, S1, measurement1, conditional CZ1, reset1 | AND1, CSwap1, Clifford20, measurement1; peak7 |
| `(3,1)` / `(1,1)` | H7, X9, CX24, T6, Tdg5, S1, measurement1, conditional CZ1, reset1 | AND1, CSwap1, Clifford18, measurement1; peak7 |

Each executes with two clean work qubits and one reusable classical slot.
AND/CSwap model categories are not additional primitive instructions. Model
T-equivalents and actual T/Tdg counts must be interpreted with their respective
expansion rules; the primitive stream has 11 T-family instructions here.

Q-0755 and Q-0756 close for the complete helper implementation and original test
file. This does not close the separate PREPARE implementation parent, all
symbolics, G2 or G3. Previously recorded permutation, Jacobi-Anger and QSP source
limitations remain open. No new child IDs or completion denominator are added.
