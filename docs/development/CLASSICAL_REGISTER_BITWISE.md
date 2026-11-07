# Classical register bitwise values

`simulation/classical_bitwise.py` owns classical integer evaluation for Xor,
XorK and BitwiseNot. Compiler-owned register modules continue to emit and count
IR; Simulation evaluates values. This module neither imports the compiler nor
selects devices or execution strategies. It introduces no foreign dtype objects
or public API exports.

The source is the unmodified Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`,
`qualtran/bloqs/arithmetic/bitwise.py`, specifically the three
`on_classical_vals` implementations. Source copyright and Apache license are
retained in the native module.

| Native entry | Original behavior retained |
|---|---|
| `xor_classical_values(source, target)` | Returns unchanged source and source XOR target |
| `xor_constant_classical_value(target, constant)` | Rejects Sympy expressions; casts constant to a NumPy scalar target's dtype before XOR |
| `bitwise_not_classical_value(target, width, unsigned=...)` | Computes -target-1; applies modulo 2**width only for unsigned/Montgomery words |

Python integers, NumPy integer scalars and integer arrays retain their numeric
types, broadcasting and source promotion/error behavior. Input arrays are not
modified. The constant-loading dtype conversion is deliberately limited to
NumPy **scalar** targets, as in the original; arrays retain ordinary XOR
promotion. Symbolic integer literals are still symbolic constants and raise.

The concrete NOT entry requires positive Python width and boolean `unsigned`.
Set `unsigned=True` for QUInt and QMontgomeryUInt value semantics. Signed and
opaque QAny value semantics use `unsigned=False`, retaining negative Python
results. Those negative classical values are not silently replaced with
unsigned representatives. A separate circuit comparison uses their finite-width
stored bit patterns. Symbolic width evaluation and upstream drawing metadata
remain outside this concrete numerical slice; the source parent is not declared
fully migrated.

NumPy unsigned scalar negation/subtraction can emit overflow warnings in the
original NOT implementation. The native implementation retains the same
operations and paired warnings. It also retains overflow/conversion exceptions
when an XorK constant cannot be loaded into a scalar dtype. Neither behavior is
silenced or changed to force a comparison to pass.

## Runnable example

```sh
python -m examples.algorithms.classical_register_bitwise
```

The example checks 5 XOR 3 = 6, and compares unsigned 6 / signed -2 complement
with `circuit.append(ComplementWord(3))`. Both classical interpretations map
stored bits 110 to 001. The actual circuit emits three X gates, with no work,
measurements or T gates. The classical evaluator is a diagnostic value model,
not a replacement for quantum simulation.

## Acceptance

G2-BIT06 adds 60 tests: 15 scalar operation/type configurations, 12 array and
broadcasting configurations, 16 exhaustive small constant/NOT word and
complex-state comparisons, 12 exhaustive XOR-pair and coherent-spectator
comparisons, one symbolic-constant boundary test and four native width/encoding
failures. Scalar tests inspect both original results and exception classes,
including signed/unsigned 8/64-bit NumPy integers. Array tests check values,
dtypes, shape, retained source identity and unchanged caller inputs. Quantum
comparisons retain the full complex amplitudes, not just output probabilities.

The concentrated bitwise/control/resource suite passes 1,174 tests without skips,
including existing exhaustive measurement branches and clean-work checks. Eight
NumPy overflow warnings occur in paired original/native scalar NOT operations.
Production and example strict typing, Black, Ruff, ownership scope, architecture
and the executable example pass. No upstream source or project dependency was
changed, and classical execution code was not added to the compiler.
