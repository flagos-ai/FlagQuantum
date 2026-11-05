# Literal measured inverses of register and constant addition

This batch follows the executable Qualtran v0.7.0 source at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`:
`Add.adjoint() -> AddK.adjoint()` and controlled AddK adjoints in
`qualtran/bloqs/arithmetic/addition.py` through the original Bloq adjoint/control
call chains. Google copyright and Apache-2.0 source notices are retained.

## Module boundaries and source components

`compiler/registers/arithmetic.py` now uses one source component sequence for
forward Add and its adjoint. It groups CNOT, And and And-adjoint components
before expanding their primitive instructions. For the inverse, component order
is reversed and And/And-adjoint are exchanged; measured primitive instructions
are not blindly reversed. Existing `add_ir` forward gate streams are unchanged.
There is no new public IR, executor, runtime allocation or reference dependency.

`add_adjoint_ir(a,b,carries,bits)` implements `b -= a mod 2**n`, with big-endian
unsigned bit patterns and 1 <= len(a) <= len(b)=n. The caller supplies n-1 clean
carry wires and n-1 distinct classical slots. It preserves a and clears every
carry. The entry is the literal Add inverse; the existing SubtractFrom
complement/add/complement source network remains a distinct decomposition of
the same mathematical permutation.

`add_constant_adjoint_ir(target,k,constant,carries,bits)` preserves the exact
`load(k) -> Add.adjoint() -> unload(k)` chain. It uses the original loaded k,
not an alternative `AddK(-k)` circuit. Constants follow the existing unsigned
contract: positive values fit n bits, negative concrete integers normalize
modulo 2**n, and booleans/symbolic values are rejected. The n constant wires and
n-1 carries start and end in zero; the target is updated in place.

`compiler/registers/controlled_addition.py` adds
`controlled_add_constant_adjoint_ir`. As in upstream controlled AddK, only
constant loading/unloading are controlled. The central Add inverse executes in
both branches; no work is skipped for inactive controls or zero constants.
Both control polarities retain their original outer X policy. Control and
arbitrary coherent target phases are preserved; all 2n-1 work wires clear.
Forward and inverse wrappers share the same normalization/load/polarity code.

## Runnable example

```sh
python -m examples.algorithms.register_inverse_addition
```

The user writes `circuit.append(ConstantAddition(k,cv,adjoint=True))`. The
example verifies every two-bit control/target amplitude with an entangled
spectator, both control polarities, positive/negative k and three work wires
returning to zero. Appending the forward operation restores the original state.
No reference package or NumPy is required at runtime.

## Actual primitive counts and source estimates

| Literal inverse path | Measured CNOT | Measured X | Measured T/TDG | Source QEC T | Measurements / source model | Clean quantum work |
|---|---:|---:|---:|---:|---:|---:|
| 8-bit Add, equal width | 81 | 0 | 28 | 28 | 7 / 7 | 7 |
| 8-bit AddK(255), plain | 81 | 16 | 28 | 28 | 7 / 7 | 15 |
| 8-bit AddK(255), control=1 | 97 | 0 | 28 | 28 | 7 / 7 | 15 |
| 8-bit AddK(255), control=0 | 97 | 2 | 28 | 28 | 7 / 7 | 15 |
| Two-bit example, k=1, control=1 | 11 | 0 | 4 | 4 | 1 / 1 | 3 |
| Two-bit example, k=-1, control=0 | 13 | 2 | 4 | 4 | 1 / 1 | 3 |

All n-bit entries emit 4(n-1) T/TDG and n-1 measurement events, independently
checked against source QEC models. The primitive set is CNOT/X/H/S/T/TDG,
measurement, conditional CZ and reset. Raw Clifford estimates are separate:
the pinned width-one Add model reports -3 Cliffords while its actual inverse
is one CNOT. This pre-existing model exception is retained, not patched or
claimed equal to a primitive count.

Gate-level parity is stronger than permutation equality. For example the
three-bit literal AddK(1) inverse emits two constant-loading X gates, while
relowering AddK(-1) emits six. Both subtract one, but only the former preserves
the requested source inverse gate stream. A regression explicitly distinguishes
them.

## Verification scope

There are 23 Add-inverse cases: 16 original source width/dtype combinations
(QUInt/Montgomery, equal/mixed widths through eight bits), six complete small
coherent permutations and validation. There are 105 constant-inverse cases:
75 original source width/constant/control combinations, 27 complete small
coherent cases, validation, the gate-flow distinction and the runnable example.
Plain, zero-control and one-control sources are all compared literally.

Source primitive operands, per-wire dependencies, feedback, T and measurements
are checked without algorithm patches. Every measurement branch for the small
Add and constant-inverse cases is postselected through actual primitive IR,
without normalization. Each equals the ideal complex state divided by
sqrt(2**(n-1)), checking phase, branch probability and all work wires. The
corresponding forward operation is checked from the computed ideal state in
every branch to prove restoration, with entangled spectators throughout.
Existing forward Add/AddK and controlled consumers are regressed together;
242 focused tests pass before the concentrated batch regression.

Symbolic inputs, signed unequal-width mathematical semantics, generalized
controls and the remaining modular algorithms are outside this slice. Final
integration CI, GPU and review remain separate acceptance gates.

Batch acceptance: 128 new and all 1,148 concentrated regression cases pass,
including pinned Qualtran/Kickmix references, with no skips. Arithmetic module
statement coverage is 98%; its two missing statements belong to the pre-existing
phase-gradient ripple helper. Controlled addition is 100%. Coverage JSON checked
against AST function spans confirms all new inverse entries and shared source
bodies are fully covered. Strict mypy passes on 944 source files; minimum/current
Black, Ruff, architecture and multi-level IR checks pass locally.
