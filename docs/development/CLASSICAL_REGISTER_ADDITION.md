# Classical addition values and quantum cross-checks

Simulation owns these numerical value rules; Compiler continues to own gate
emission and resource declarations. The source is unmodified Qualtran v0.7.0
commit `bcb17bb25360ed68ce3bc084645db3e929799389`:
`simulation/classical_sim.py:add_ints`, arithmetic `addition.py` Add/AddK/
OutOfPlaceAdder, and `controlled_addition.py:CAdd.on_classical_vals`.
Copyright and Apache source attribution are retained. Production imports no
Qualtran dtype, compiler helper, execution strategy or device interface.

## Ownership and interfaces

`simulation/classical_addition.py` provides the shared `add_integer_values`:
add first, then optionally wrap modulo 2**num_bits; signed results use the
source's centered interval. Omitting width retains ordinary arbitrary-precision
addition. Operand NumPy types/promotion remain unchanged at this shared layer.
Its consumers retain their original distinct policies:

| Entry | Conversions and port behavior |
|---|---|
| `register_add_classical_values` | Preserves left object; converts both operands and target width with Python int |
| `add_constant_classical_value` | Rejects symbolic width or constant, including Sympy integer literals; converts target/constant to Python int |
| `out_of_place_sum_classical_values` | Preserves a/b, creates c from integer operands; optional carry bit; adjoint removes supplied c |
| `controlled_add_classical_values` in `classical_controlled_addition.py` | Preserves ctrl/a; inactive returns original b before symbolic-width validation; active uses shared arithmetic without operand casts |

The caller supplies concrete width and signed/unsigned interpretation from its
register contract. Left logical width need not be passed to the value model:
upstream Add/CAdd's direct classical method uses only the target's width. These
functions model the original classical methods, not all dtype domain checks of
upstream `call_classically`. Caller-owned register objects are not replaced by
foreign SDK objects. NumPy scalar/array errors and integer-conversion warnings
are retained rather than normalized away.

Ordinary Add can evaluate a Sympy Integer target width through int conversion;
AddK rejects the same symbolic literal. CAdd accepts an unresolved symbolic width
on an inactive branch, preserving the exact array objects, but rejects it on the
active branch. These distinctions are tested directly against unmodified source.

## Boundaries that remain distinct

OutOfPlaceAdder's classical adjoint checks c is present but does **not** verify
c=a+b. The native classical model likewise removes an incorrect supplied c.
Forward c presence and inverse c absence retain explicit AssertionError, matching
source normal-mode assertion contracts. This is diagnostic source behavior,
not a quantum clean-erasure certificate. Actual measured erasure tests provide
only the correct computed sum and verify every measurement history.

The existing unequal-width signed CAdd discrepancy remains open. For
CAdd(QInt(1), QInt(2)), ctrl=1, a=-1, b=0, the source and native classical models
return -1. The literal source quantum chain and existing native emitter
zero-extend the stored one-bit pattern and return +1. The regression retains
both values; it does not silently introduce sign extension or claim parity on
that conflicting domain. Same-width signed words and unequal-width unsigned
words are cross-checked coherently on every small input.

## Runnable append example and resources

```sh
python -m examples.algorithms.classical_register_addition
```

The example checks the three ordinary value models, then uses
`circuit.append(ControlledWordAddition())` and 16 actual trajectories. Control 1
and source 3 are preserved; two-bit target 2 becomes 1 and all three work wires
return to zero. Exhaustive unit tests establish the branch semantics separately.

| Quantity | Actual instructions | Independent logical model |
|---|---:|---:|
| T + TDG | 12 | 3 And converted to 12 T |
| Measurement | 3 | 3 |
| Total qubits / work | 8 / 3 | Peak 8 |
| Clifford | 9 H, 21 CX, 3 S, 3 conditional CZ | 6 Clifford, leaving And separate |
| Reset | 3 | Included in measured erasure semantics |

No model T-equivalent replaces an actual primitive scan. The classical helpers
emit no gates and make no separate quantum-resource claim.

## Acceptance

G2-A51 adds 103 tests for shared scalar/array arithmetic, full small signed/
unsigned/Montgomery classical domains, Python conversion policies and failures,
OutOfPlaceAdder port direction, source symbolic boundaries, full complex Add
branches and correct-sum measured cleanup. G2-A52 adds 37 tests for both CAdd
polarities, every small classical input, inactive symbolic array identity,
active NumPy promotion, full coherent controls/inputs and all work-erasure
histories, plus the preserved signed discrepancy.

The combined suite passes 510 tests without skips. Four paired array-conversion
deprecation warnings and ten paired scalar-overflow warnings are retained across
new addition and existing bitwise cases. Three production/example files pass
strict typing; Black, Ruff, Simulation scope, architecture and the executable
append example pass. Compiler implementations and pinned references are
unchanged. Parent source entries and G2 remain open.
