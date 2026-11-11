# Component tensor providers (GRAPH81)

`simulation/tensor_network/component_tensor_providers.py` binds exact Core
function names to numerical leaf suppliers. It keeps partial expansion and
numerical dispatch on one explicit mapping, so a component retained by the
Compiler is the same component the Simulation supplier can handle.

```python
providers = ComponentTensorProviders({"hadamard_leaf": provide_hadamard})
selected, order = expand_components_for_tensor(
    program, providers.has_custom_tensor, full_flatten=False
)
matrix = contract_component_tensors(selected, order, providers.supply, like)
```

The mapping is copied at construction and exposed only through read-only lookup.
It may be used with either the three-argument pure supplier or the four-argument
mixed supplier; the mixed form receives the existing fresh-label callback.
Provider names are exact `FunctionRecord.name` keys. No Python method inspection,
global registry, guessed fallback, automatic decomposition or simulator choice
is performed. A missing selected provider raises `ValueError`, and supplier
exceptions and yielded trace tokens continue through the existing contracts.

Qualtran v0.7.0 `bloq_has_custom_tensors` instead detects whether a Bloq instance
overrides `my_tensors` and invokes that bound method during traversal. Core
`FunctionRecord` does not carry a Python Bloq instance or method implementation,
so importing or imitating that name heuristic would be an unsafe inference.
This exact mapping is FlagQuantum's explicit counterpart: callers define the
function-to-supplier relationship once and pass the same leaf predicate and
supplier into the existing Compiler and Simulation owners.

Tests verify that the same binding drives selective expansion and pure dense
contraction against the original Bloq tensor, that caller mutation of the input
dictionary does not change the mapping, exact keys do not use case/name
heuristics, invalid keys/values are refused, missing entries fail explicitly,
callback exceptions preserve identity, and a mixed provider receives fresh
labels and matches the original channel. The new owner has complete statement
and branch coverage. This remains an internal Simulation helper, not a stable
public export or a global provider registry.

## Acceptance evidence

Implementation `7f940717` adds seven tests, including a replacement supplier
that emits a scaled leaf and a compensating scalar without changing the
consumer; its full matrix and parameter gradient agree. Final related regression
passes 1489 cases in 44.08 s, focused verification passes 764 in 40.74 s, and
64 integration cases pass in 6.47 s. The owner covers all 23 statements and four
branches. Strict mypy passes 44 related files plus three source-following files;
Black/Ruff, Simulation scope, architecture, Program and protected API checks pass.
Thirty-two locked source blobs match the inventory, 25 unchanged upstream tests
pass in 1.14 s, and the native Kickmix Simulator and binary-field append example
are checked. Existing measurement-branch and clean-work regressions remain in
the related suite; dispatch is not a new physical decomposition.

Provider lookup adds zero physical gates or work bits and changes no resource
model. It retains existing tensor suppliers and contraction owners. The pure
matrix, mixed channel, shared fresh environment labels and supplier replacement
are numerical checks; this batch makes no new physical gate-count parity claim.

Default smoke/unit stops at maxfail with five failures, 21400 passed, 300 skipped
and 1037 warnings in 199.58 s. All five failure IDs belong to the independent
seven-case baseline; two were not observed. There is no new observed failure,
but the default suite is not green or complete. Logs, resources, source hashes
and failure comparison are under `ftqc-reference-env/g2-component-tensor-providers-*`.
