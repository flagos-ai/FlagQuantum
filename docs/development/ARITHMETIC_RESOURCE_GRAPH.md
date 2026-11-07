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

## Implemented source recipes

`arithmetic_callees` retains the direct Add, CAdd and four arithmetic shim
recipes, including source insertion order and adjoint-wrapper behavior.
`arithmetic_call_graph` expands these plus BitwiseNot's OnEach(X) into the
existing typed QEC leaves. Word encodings remain on arithmetic keys; OnEach
uses the existing width/child cost key, which canonicalizes its register dtype.
This is a logical resource graph, not a serialized SDK object graph.

The seven category queries match the original source for Python and symbolic
widths, both cost profiles and both directions. Raw negative counts, one-bit
Add's declared negative Clifford term and zero-width BitwiseNot failures remain
visible. No width-lifetime estimate is inferred from the call graph: concrete
and symbolic width queries continue using the separate existing formulas.

Run an end-to-end symbolic query:

```sh
python -m examples.algorithms.arithmetic_resource_graph
python -m pytest -q tests/unit/test_arithmetic_declarations.py
```

At n=3, CSub declares 5 And, 23 Clifford, 5 measurements and 20 terminal T
equivalents. These model values are not an emitted primitive-gate report.
The modular raw declaration API now delegates to these native recipes for both
concrete and symbolic widths. Aggregate estimate results remain unchanged.

## Validation evidence

The batch adds 35 identity/replacement tests and 77 recipe/reference tests.
The concentrated regression has 635 passing tests with no skips, including the
existing concrete Add/CAdd full-state, measurement-feedback and clean-work tests.
Production modules and the example pass strict mypy, formatting, lint and the
architecture boundary check. This establishes the seven supported logical-call
recipes and their cost queries, not full Qualtran coverage or physical execution
of the source shims. Existing concrete aggregate APIs are unchanged.

## Formula consumers (G2-RES38)

`source_controlled_add_resource_formula` now queries the declared CAdd edges for
symbolic dimensions, with shared Add and predicate model costs supplied through
existing static overrides. `source_modular_arithmetic_resource_formula` queries
all four native placeholder recipes for both concrete and symbolic dimensions.
Their width-lifetime calculations remain independent. `modular_arithmetic_calls`
uses the same recipes and returns unfurled native keys, replacing its previous
concrete-only folded costs; it no longer owns duplicate edge formulas.

Replacement tests prove both recipe and model substitution without consumer edits.
The source zero-width CSub direct declaration is now inspectable, while its
failing recursive/aggregate resource query remains explicit.

The consumer batch passes 683 concentrated tests without skips (48 net new
cases), strict mypy and architecture checks. Both existing modular examples and
the symbolic call-graph example remain runnable.
