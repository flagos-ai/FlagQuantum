# Tensor flattening source disposition

Reference: unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, complete
`qualtran/simulation/tensor/_flattening.py`. There are exactly two functions,
no classes, constants, assets or additional initialization. Apache-2.0 source
attribution remains in the existing Compiler transformation owner.

## Interfaces and ownership

| Source surface | Existing native owner | Disposition |
| --- | --- | --- |
| `bloq_has_custom_tensors` | Simulation `ComponentTensorProviders.has_custom_tensor` | Explicit exact function-name declarations replace foreign Python `my_tensors.__qualname__` inference. The same immutable map supplies selected leaves. Classification never executes a supplier. |
| `flatten_for_tensor_contraction` | Compiler `component_tensor_expansion` | Explicit composite and root entries reuse `expand_component_calls`. Root wrapping, full/partial selection, traversal, persistent occurrence labels and typed SSA wiring are already implemented. No new executor or transformation is necessary. |

Compiler consumes only the existing name predicate; it does not import Simulation
or run numerical callbacks. Simulation receives the selected Core program and
operation positions. Persistent occurrence labels are not positions. Registers
retain their declared direction, shape, row-major word and MSB-first bit order;
transformation creates no physical work or numerical label ownership.

## Complete heuristic and failure audit

The new source-backed tests invoke the original functions without replacing their
bodies. Source classification rejects methods whose qualified name starts with
`Bloq.` or `GateWithRegisters.` and accepts custom and inherited overrides.
`functools.wraps(Bloq.my_tensors)` preserves the base name, so the source rejects
that wrapper even though it is a different callable. Native explicit declarations
are independent of callable names and spelling aliases; this is an intentional
architectural replacement, not a claim of identical foreign reflection.

Source full flatten always attempts decomposition; partial flatten avoids it
when the source classifies the leaf as custom. The delegated `flatten_once`
retains leaves after `DecomposeTypeError` or `DecomposeNotImplementedError`.
Ordinary `RuntimeError` and `ValueError` propagate with object identity. Tests
cover the full/partial, custom/noncustom and four-exception cross product,
including exact callback count and persistent instance zero for retained roots.

Native typed definitions are materialized before this boundary. A primitive body
remains a leaf; an empty typed-call body is a valid decomposition and disappears.
There is no foreign lazy factory here and no error conversion into such a factory.
Classifier exceptions propagate unchanged, including source decomposition-error
classes supplied as callback failures. In full mode the classifier is never
called. Invalid native graphs are refused at Core validation, before selection.
This is stricter than the foreign object's lazy evaluation and remains an
explicit difference; no broad exception catch or foreign dependency is added.

A separate empty-identity test verifies that zero-call decompositions disappear
in both paths, and that the existing native idle-identity wiring produces the
same complete complex matrix without invoking a supplier. Existing root tests
cover unknown/untyped/invalid entries, name collisions, empty signatures and
depth exhaustion. Existing component expansion tests cover source traversal,
labels, unchanged definitions, complex phase/gradient, physical gate order and
small measured-Add history pairs with restored clean work. No additional quantum
algorithm is introduced by this audit.

## Acceptance and remaining scope

The complete original `_flattening_test.py` already runs unchanged and with its
classification boundary replaced in `test_tensor_protocol_source_audit.py`.
This batch supplies missing reflection and lazy-failure disposition, not another
copy of the production implementation. Q-0889 remains open under closure rule5:
foreign reflection and deferred decomposition callbacks are intentionally absent,
and their arbitrary object/error paths are not claimed as supported native APIs.
No source defect is repaired or reference exception changed to claim equality.

Actual added primitives, measurements, feedback slots and work bits: **zero**.
Quantum model estimate: **not applicable** for a graph selection audit. Existing
physical Add gate counts and source QEC estimates remain separate. Numerical
test tensors and graph traversal are not quantum gate costs or capacity evidence.
The historical default-suite timeout remains open and is not rerun unchanged.

Next: Q-0891 `_quimb.py` zero-initialization/discard source boundaries, audited
against existing Compiler initialization and pure/mixed node owners before any
production change. Do not repeat width/ADCR or foreign reflection infrastructure.

Local acceptance: **23 new scenarios**, **2,502 related tests in 50.66 s**,
**84 CPU/TN tests in 5.36 s**, strict mypy, Black/Ruff, Simulation scope,
architecture, IR, protected API and executable Bell example pass. Four frozen
source blobs match the inventory. Acceptance commit `e6ad2789`. Existing source
warnings and default-suite timeout remain; this is not full-suite certification.

Coverage for the existing tensor-expansion and explicit-provider owners is
**45 statements and 10 branches at 100%**. This is bounded owner coverage,
not certification of all Compiler transformations or foreign object types.
