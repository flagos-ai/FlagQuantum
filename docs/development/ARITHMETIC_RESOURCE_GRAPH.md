# Arithmetic resource declarations

Register recipes own arithmetic decomposition declarations; `_component_trace`
owns immutable keys, graph traversal and cost queries. Neither executes circuits.
The current consumers are controlled addition and the modular subtraction shim.
Users of executable operators still use `circuit.append(op)`.

`ComponentCost` represents concrete resource counts and cannot express symbolic
operand identities. `SymbolicResourceFormula` represents aggregate categories,
not callees. `ArithmeticCall` therefore records only the operation, unchanged
width objects, word encodings, control polarity, signed flag and direction.
Recipes live in `compiler/registers/arithmetic_declarations.py`; they supply the
existing `DeclaredResourceCalls` interface to existing graph and QEC queries.
No SDK objects, registry, backend selection or numerical execution enter this path.

Keys distinguish Python integer widths from Sympy expressions because the pinned
source dispatches decomposition differently for them. The upstream objects compare
Python `3` and Sympy `Integer(3)` equal; this native cache safeguard deliberately
does not reproduce that equality. Reference queries use separate caches and retain
the original width objects. Callee parameters and resource formulas are compared
against unmodified Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

An arithmetic key without a declaration fails QEC and terminal-T queries rather
than silently contributing zero. Declaration maps can be replaced without changing
query consumers; contract tests exercise this with alternate leaf implementations.
Raw negative/zero source multiplicities remain diagnostic declaration data.
Executable `CostCall` validation remains nonnegative. The source modular shims
have no gate decomposition: declared estimates do not establish executable parity.

Run the shortest contract checks with:

```sh
python -m pytest -q tests/unit/test_arithmetic_resource_identity.py
```
