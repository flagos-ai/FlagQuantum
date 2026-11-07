# Classical subtraction and typed negation

Simulation owns the numerical values and their validation. The existing compiler
emitters remain unchanged. The pinned unmodified source is Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`: arithmetic `subtraction.py`,
`negate.py`, the already migrated BitwiseNot/AddK leaf value rules, and integer
encoding validators in `_infra/data_types.py`. Source copyright and Apache
attribution are retained; production imports no Qualtran objects.

## Modules and interfaces

`simulation/classical_subtraction.py` retains two source policies:

- `subtract_classical_values(left, right, width, signed=...)` preserves left and
  computes left-right. The signed flag corresponds to the original **left**
  dtype. Subtraction and centered modulo happen before Python integer conversion.
  NumPy underflow, promotion, array conversion errors and warnings remain visible.
- `subtract_from_classical_values` preserves left and computes right-left. The
  source casts both operands and width to Python integers before using the shared
  addition model. These conversions are not moved into Subtract's earlier stage.

Sympy literal widths also retain source result types: signed Subtract can return
a Sympy integer after subtracting its symbolic half-modulus, while SubtractFrom
returns a Python integer. Unresolved width failures keep their original exception
classes. These are direct value-model semantics, not the full dtype-validation
protocol of `call_classically` or a claim of executable symbolic gates.

`simulation/integer_values.py` owns `validate_integer_word_value`: positive,
concrete scalar integer-word bounds without coercion. Python integers, booleans
and NumPy integer scalars follow original signed/unsigned bounds. Arrays and
noninteger scalar objects are rejected for scalar registers. Stored Montgomery
values use the same width-bounded unsigned validation. At this batch's acceptance only positive concrete scalar widths were covered.
[G2-VAL01](CLASSICAL_REGISTER_VALUES.md) subsequently extends the shared validator
to zero/symbolic word widths and array ranges. The full foreign dtype hierarchy
remains outside this helper; Negate still requires its concrete word contract.

`simulation/classical_negation.py` composes the existing native value models in
source order: validate input -> BitwiseNot -> validate intermediate -> AddK(1)
-> validate output. It does not replace the call chain with unary minus. This
retains a source failure when unsigned NumPy scalar negation produces an invalid
signed intermediate, even if a later wrap could have hidden the error. Stage
replacement tests prove consumers keep this composition.

## Source boundaries

Zero-width Negate cannot bind its x input upstream and raises TypeError; the
native composition exposes that failure. Symbolic widths, including integer
literals, remain rejected. One-bit signed Negate has a valid classical result
(-1 maps to -1) but the pinned gate decomposition cannot load signed +1 in that
word. The existing native measured circuit remains verified separately; this
batch does not repair or claim parity with that failing source decomposition.

Subtract's direct classical model and typed Negate composition intentionally
have different validation boundaries because the source protocols differ. Values
are never silently recast to merge the protocols. No compiler code executes
numerical values, and no classical result alone proves quantum work cleanup.

## Example and resource accounting

```sh
python -m examples.algorithms.classical_register_subtraction
```

The example checks 1-3 modulo 8 = 6 and 3-1 modulo 8 = 2. It then uses
`circuit.append(NegateWord())` to map stored 010 to 110: unsigned 2 becomes 6,
or signed 2 becomes -2. Sixteen actual trajectories check all five work wires
are zero; exhaustive branch evidence comes from the tests.

| Three-bit negation | Actual instructions | Independent source model |
|---|---:|---:|
| T + TDG | 8 | 2 And -> 8 T |
| Measurements | 2 | 2 |
| Total qubits / work | 8 / 5 | AddK peak 8; complement uses only target |
| Clifford | 5 X, 6 H, 21 CX, 2 S, 2 conditional CZ | AddK 13 Clifford + three complement X |
| Reset | 2 | Measured erasure semantics |

The printed AddK model is explicitly the shared child model; the complement's
three X gates are listed separately. Primitive scans and logical estimates are
not interchanged.

## Acceptance

The batch adds 90 tests: 54 for both subtraction directions and conversion
boundaries, 30 for typed Negate and stage replacement, and six for scalar integer
validation against three original dtypes. Tests cover full small classical
inputs, signed unequal-width subtraction, NumPy values/types/errors, symbolic
literal distinctions, complete complex amplitudes with spectators, every small
measured history, prefix/carry/constant cleanup and negation restoration.

The combined arithmetic suite passes 426 tests without skips, including existing
original gate-stream/resource tests. Twenty-eight source/native paired NumPy
conversion/overflow warnings remain unsuppressed. Four production/example files
pass strict typing; Black, Ruff, Simulation ownership, architecture and the
executable example pass. Parent files, known source failures and G2 remain open.
