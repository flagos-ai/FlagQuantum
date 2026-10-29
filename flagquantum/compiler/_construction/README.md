# Coherent integer construction profile

Compiler owns this private profile of the existing `_hybrid.model` records.
It verifies coherent integer expression definitions and physical binding
preconditions. It does not capture Python, synthesize gates, select candidates,
run circuits or export a public `synth` API.

The same `IRType`, `ValueId`, `Value`, `Operation`, `Block`, `Region`, source
location and `HybridProgram` are reused. `quantum.uint(width)` represents a
logical basis-encoded value, not an allocated register or a consumed effect
token. Function inputs are immutable basis borrows. Repeated reads of a value
are legal; overlapping physical input/output/workspace ports are not.

Supported records are `arith.constant` with classical INDEX type and a checked
nonnegative integer, `qint.add`, `qint.multiply`, `qint.call`, and `qint.return`.
Functions return one quantum integer expression. Definitions form an acyclic
call graph. This is a closed verifier profile, not an extension to the existing
classical hybrid verifier or its lowering pipeline.

An eventual lowering must implement `z ^= f(inputs)` for every target bitstring
with zero workspace, preserving phases and cleaning workspace. This code checks
definitions and bindings; it does not establish that an arithmetic lowering
exists. `fresh_zero` is an allocator obligation; `caller_asserted_zero` is a
caller precondition. Neither is inferred from an external state.

For a small change, edit the corresponding verifier, add an accepted/rejected
scenario, and run:

```bash
python -m pytest tests/team/compiler/test_construction_profile.py -q
python -m pytest tests/team/compiler/test_construction_dependencies.py -q
python -m pytest tests/integration/test_construction_execution_boundary.py -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```

The integration probes use existing Circuit/IR/runtime capabilities. They do
not report arithmetic synthesis as implemented. The next step is CS-02's pure
adder-kernel extraction and conservative lowering behind this boundary.

See the [CS-01 contract](../../../docs/api-changes/FQ-SYNTH-CS01-20261029.md)
and [Classiq review](../../../docs/roadmap/CLASSIQ_SEMANTICS_REVIEW_20261029.md).
