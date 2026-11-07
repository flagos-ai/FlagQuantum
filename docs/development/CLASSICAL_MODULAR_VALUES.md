# Classical modular register values

`simulation/classical_modular_addition.py` ports pinned ModAdd and CModAdd
`on_classical_vals` into numerical functions, without compiler dependencies.
The plain function preserves the source exactly-one-equals-modulus exemption,
comparison order, NumPy types and new-array arithmetic. The controlled function
returns inactive operands by identity before any domain or modulus checks, then
reuses the same addition evaluator on active branches. Both control polarities
and the original behavior for other control values are retained.

The source exemption is broader than its documented valid domain: when one word
equals the modulus, it can admit an otherwise invalid other word. These outcomes
are retained as numerical diagnostics, not advertised as clean quantum execution.
The existing y=modulus gate cleanup failure remains open. No source reference is
modified, no normalized modulo expression replaces the original checks, and no
new register or graph contract is introduced.

Input types, symbolic relations, missing residue bounds, zero modulus, unsigned
scalars and array truth failures are compared to the original methods. Direct
value-method tests are distinct from typed register-domain validation and from
quantum gate-flow evidence.


## Modular subtraction and controlled subtraction

`simulation/classical_modular_subtraction.py` preserves original ModSub/CModSub
comparison and operation order. Outside `[0,p)` both operands are returned
unchanged. Active subtraction uses the original `right -= left`, then adds the
modulus only if the result is negative. Single-element arrays therefore retain
in-place mutation and caller aliases; unsigned scalar underflow is not repaired.
Controlled subtraction checks activation before entering the shared evaluator.
These are direct value semantics. They do not broaden the compiler's valid clean
residue domain, or make arbitrary NumPy values valid quantum words.

The source CModAdd and CModSub methods are compared for both polarities, inactive
values and non-bit controls; the method itself does not add a control validator.
Typed control validation remains distinct. The plain and controlled coherent
state builders now use these evaluators while retaining independent mathematical
assertions. Existing source primitive/feedback/resource checks remain unchanged.
Small forced measurement histories and all-history clean-work certificates stay
in the quantum suites, including the known y=p addition and p+1 load boundaries.

Run the example without upstream reference packages:

```sh
python -m examples.algorithms.classical_modular_values
```

The example uses `circuit.append(op)` on a coherent two-bit modulo-2 pair and
compares complete trajectory amplitudes after addition and subtraction. Both
operations restore six work wires. Emitted ModAdd is 28 T and 7 measurements;
ModSub is 36 T and 9 measurements. The corresponding pinned model values are
listed separately, using `16*n-4` and `24*n-12` T-equivalent formulas at n=2.
This example executes eight sampled trajectories; exhaustive small histories and
larger all-history certificates are provided by the acceptance tests.

Acceptance: 626 new classical cases (313 per module), 766 concentrated passes,
no skips. Nineteen source/native unsigned arithmetic overflow warnings remain
visible, including the explicit native underflow witness. Strict types, formatting,
team scope, architecture and the runnable example pass. Original source blobs
are verified against the inventory; no reference or prior defect is patched.

## Modular negation value methods (G2-VAL07)

`simulation/classical_modular_negation.py` ports ModNeg/CModNeg with the source's
strict interior test `0 < x < p`. Zero, negative and at-or-above-modulus values
return unchanged, and controlled evaluation tests activation before comparing x.
No control validator or modulo normalization is added. The original gate cleanup
failure outside the clean residue domain remains open.

The existing basis/coherent state builders now consume this evaluator while
retaining the independent mathematical negation assertion. Original primitive,
feedback, QEC/QubitCount, forced-history and all-history clean-work assertions are
unchanged. This module adds 217 source value tests covering types, symbolic
relations, array truth behavior, exterior identities and both control polarities.
