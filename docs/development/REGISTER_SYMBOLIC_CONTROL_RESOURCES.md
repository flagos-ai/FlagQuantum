# Symbolic control shapes and resources

`control_predicate_resource_model.py` estimates known activation tuples or an
unknown-polarity count through existing `ControlPredicateCall` identities.
`controlled_component_resource_model.py` composes the positive child's model,
predicate and measured erasure through existing QEC and width queries. Concrete
`ControlRegister` consumers delegate to the same pattern query; no parallel
predicate compiler or SDK-shaped object has been added.

The unmodified reference is Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, specifically CtrlSpecAnd,
ControlledViaAnd, Register shape iteration and QubitCount. Source copyright and
license remain attributed in the implementation. The existing concrete control
emitter and its explicit work ownership are unchanged.

## Interfaces and the shape distinction

`control_predicate_resource_formula(controls, uncompute=..., legacy_shims=...)`
uses the existing `ControlPattern`: a binary activation tuple, a Python integer
for an unspecified activation of known length, or an unchanged symbolic integer
expression. It returns all seven source QEC categories and source peak width.

`controlled_pattern_resource_formula` accepts a positive resource key/model,
logical target width, control pattern and optional `control_shapes`. Shapes are
immutable tuples of register dimension tuples, in original control-port order.
They describe the original register shapes; they are not a request to select a
cheaper estimate. A caller supplying count rather than activation values owns
the total stored-bit count, including word widths and shape products. No source
QDType or Shaped object crosses into FlagQuantum production.

- Concrete shape dimensions allow the original outer graph to be constructed,
  even when control **word widths** are symbolic. The existing connection-width
  scheduler accounts for predicate storage and both live data words for XOR.
- Symbolic shape dimensions prevent source register-index iteration. The original
  whole-wrapper fallback takes the signature/callee maximum without executable
  outer bystanders. It folds positive body, compute predicate and erase predicate
  in declaration order, matching the source and Sympy reduction order.
- Without explicit shapes, a scalar count describes one bit array with that
  count as its dimension. Concrete activation tuples describe concrete storage.

This distinction differs from a symbolic AddK **inside** a concretely shaped
outer wrapper: inner fallback does not remove the outer lifetime connections.
The two mechanisms now share one authority while retaining different dispatch.

| XOR with three-bit source and target | Source peak |
|---|---:|
| Unknown activation, Python count 2 | 9 |
| Unknown activation, Sympy Integer(2) array shape | 8 |
| Symbolic m-bit array, then substitute m=3 | 9 |
| One symbolic m-bit word, then substitute m=3 | 11 |

The equivalent AddK(width=3, constant=5) Python/Sympy two-control peaks are 11/9.
These differences are not removed by casting parameters or flattening away shape
metadata. The numerical equations remain separate from gate execution.

## Consumers and source boundaries

`source_pattern_controlled_xor_resource_formula` in `xor_resource_model.py` and
`source_pattern_controlled_constant_add_resource_formula` in
`grouped_constant_add_resource_model.py` supply their existing positive models
to this shared query. Their optional `control_shapes` preserve scalar-word vs
array semantics. Existing concrete grouped APIs remain usable as before.

Unknown-polarity single-control wrappers fail upstream because the activation
bit cannot be read; native consumer queries reject them too. Python zero control
count is rejected. A symbolic-zero predicate can return negative declarations;
these are retained as diagnostics and must not be used as physical budgets.
Symbolic counts and shapes never produce executable instructions or a claim
that unknown work wires can be allocated. Explicit one-bit activation tuples
`(0,)` and `(1,)` retain their working concrete semantics.

Run `python -m examples.algorithms.arithmetic_resource_graph` to inspect the two
symbolic control-shape paths. The append workflows remain
`grouped_constant_xor` and `multi_controlled_constant_addition`; they use concrete
registers and caller-owned clean work.

## Acceptance

G2-CTRL02 adds 505 tests: 44 standalone predicate queries and 440 two-consumer
queries across forward/adjoint, both QEC profiles, concrete/symbolic data widths,
symbolic constants, multidimensional shapes and symbolic integer control words.
All seven categories and QubitCount compare against unmodified source. Two
witnesses distinguish literal and shape dispatch, six cases retain source
zero/one failures, seven reject invalid patterns and four reject invalid shapes.
Two shared-predicate replacements carry all seven categories and larger private
peaks through both width-dispatch paths without changing consumers.

The concentrated suite passes 1,782 tests without skips, including existing
source gates, every small coherent measurement history, clean work and consumer
replacement regressions. Six existing LCU reference numerical warnings remain
visible. Five production/example files pass strict typing, and Black, Ruff,
ownership scope, architecture and the model example pass. Source shape and
failure boundaries remain documented; this batch does not complete G2 or the
whole upstream control infrastructure.
