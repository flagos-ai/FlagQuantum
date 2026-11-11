# Component tensor providers (GRAPH81)

`simulation/tensor_network/component_tensor_providers.py` binds exact Core
function names to numerical leaf suppliers. It keeps partial expansion and
numerical dispatch on one explicit mapping, so a component retained by the
Compiler is the same component the Simulation supplier can handle.

```python
from flagquantum.compiler.registers.component_graph import iter_component_calls

providers = ComponentTensorProviders({"hadamard_leaf": provide_hadamard})
selected, instance_ids = expand_components_for_tensor(
    program, providers.has_custom_tensor, full_flatten=False
)
positions = tuple(i for i, _, _ in iter_component_calls(selected))
matrix = contract_component_tensors(selected, positions, providers.supply, like)
```

`instance_ids` preserves upstream occurrence identities for source comparison.
It is not the positional visitation order consumed by Simulation. Recompute
positions using the existing Compiler `iter_component_calls` after expansion;
for a wrapped root, pass its entry name to both traversal and contraction.

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

## Expanded graph visitation (GRAPH82)

The expansion result's second tuple contains persistent occurrence IDs, matching
Qualtran `BloqInstance.i`. These are provenance labels, not offsets into the
new entry block. For example, expanding two calls can produce IDs `(2, 3, 4, 5)`
for operations stored at positions `(0, 1, 2, 3)`. Simulation consumes the latter
in the traversal order supplied by `iter_component_calls`. Its order validation
refuses the former before any supplier runs. This distinction also applies to
`expand_component_root_for_tensor`, including the terminating no-change probe.

The example above now derives positions after expansion. Previous GRAPH81
examples passed returned IDs directly; they happened to work for retained
atomic leaves because those IDs equaled their positions. The implementation's
two distinct contracts are unchanged; this corrects their composition in the
example and its conformance tests. It does not repurpose source instance IDs.

Twenty-four new cases exercise multi-level full/partial expansion, composite
and wrapped roots, one/two calls per level, pure complex matrices and mixed
retained/discarded channels. They compare persistent IDs and each pure leaf's
data with the unchanged source, then contract using computed positions. A
source-backed replacement supplier preserves the same mixed consumer. Complex
relative-phase derivatives, spectator coherences and the existing append stream
are checked. Sixteen pure scenarios execute the bound operation with unused
clean storage. Eight mixed scenarios execute both classical idle inputs and
all four joint outcomes: 64 branch executions compare complete complex
amplitudes after the documented X-environment frame restoration. That frame
change is only a numerical comparison; it adds no physical H.

Pure physical graphs emit `repeats**depth` H instructions, matching the source
Clifford model for the selected 2/3-level cases. Each mixed graph emits H1,
measurement2 and classical entry XOR1, with zero reset/T; its source QEC model
reports measurement2 and Clifford0/T0. This difference remains explicit: the
source X tensor has no corresponding physical decomposition, and classical
entry initialization is a native adapter cost. Graph selection and corrected
position traversal themselves add zero gates or work bits. Returned numerical
storage and source QubitCount remain distinct from physical clean storage.

GRAPH82 evidence (`0fc1574f`): 24 new cases, 1513 related tests passed in 31.08 s,
788 focused tests in 30.51 s, and 64 integration tests in 4.81 s. Strict typing
passes 45 related files plus three source-following files. Black/Ruff, Compiler
and Simulation scope, architecture, Program and protected API checks pass.
Thirty-two source blobs match the locked inventory; 25 unchanged original tests
pass in 1.45 s. Native Kickmix Simulator presence and the binary-field example
are checked. Eighteen related owners cover 556 statements and 214 branches at
99%, retaining the historical two GRAPH60 profile-refusal misses. No new
production branch or physical instruction is introduced by this correction.

Default smoke/unit stops at maxfail: six failed, 21485 passed, 300 skipped and
1037 warnings in 139.42 s. All six failure IDs belong to the independent
seven-case baseline; one was not observed. There is no new observed failure,
but the default suite is not green or complete. Logs, source hashes, resources
and failure comparison are under `ftqc-reference-env/g2-expanded-tensor-providers-*`.
