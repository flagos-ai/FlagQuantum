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
