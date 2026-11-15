# Block encoding rewrites

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Frozen source | Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0294 | `rewrites.py` | `209c5aad537c9cbd8458d26294bdb6eb09a5a43d` | Open for generic foreign term identity and custom-oracle rewrite semantics. |
| Q-0295 | `rewrites_test.py` | `79422a95c589b573eea88b689686508b69cd9ab1` | Closed. Both original test scenarios have native equivalents. |

The rewrite operates on unbound Algorithms block objects. It traverses
Phase, Product, and TensorProduct and accumulates nested default LCU terms
in first occurrence order. A nested term's coefficient is multiplied by
its parent's coefficient before accumulation. The output precision is the
maximum of the parent precision and the parent-plus-child value for each
nested LCU. Cancellation preserves zero-valued terms. A single remaining
unique term retains the source constructor's minimum-arity refusal.

Native emitters are Python callables and their existing equality and hash
govern term equivalence. The source's separate structurally equal X gate
objects can merge. Two independently constructed native callable wrappers
remain distinct unless their existing equality establishes equivalence.
The test reuses the same native X object for the original duplicated term.
When presented with explicit PREPARE or SELECT overrides, FlagQuantum keeps
that block opaque and preserves its oracles. The source rule rebuilds LCU
objects with default oracles. No generic callable or external-block
compatibility contract is introduced.

The original atomic/phase/product scenario and the nested tensor product
scenario both pass. The latter changes the nested terms to X, H and phased H
with coefficients 1, 2 and 1 at precision two, keeping the outer tensor
factor. The numerical test binds the rewritten three-term child through the
existing append path. Model and runtime evidence refer to this rewritten
child and retain the packed system, selection, coefficient-junk, and temporary
ownership. All 2,048 measurement histories match the complete reference
complex output for the prepared system input. Each history also matches the
projected block and clears all five temporary work wires.

| Quantity | Physical emission | Independent source model |
| --- | --- | --- |
| T/T-dagger or model terms | 72 T/T-dagger | Four Fredkins and eleven And operations |
| Measurements | 11 | 11 |
| Width | 15 reserved | 15 peak logical qubits |
| Additional modeled categories | Reported as primitive instruction opcodes | 110 Cliffords and four rotations |

```python
from examples.algorithms.product_block_catalog import product_child
from flagquantum.algorithms.primitives.block_encoding_rewrites import collect_like_terms
from flagquantum.algorithms.primitives.linear_combination_block_encoding import (
    LinearCombinationBlockEncoding,
)

x, h = product_child("x"), product_child("h")
left = LinearCombinationBlockEncoding((x, h), (2.0, 2.0), 1)
rewritten = collect_like_terms(LinearCombinationBlockEncoding((left, x), (1.0, -1.0), 1))
assert isinstance(rewritten, LinearCombinationBlockEncoding)
assert rewritten.coefficients == (1.0, 2.0)
assert rewritten.coefficient_bits == 2
```

The complete append and source-model checks are in
`tests/unit/test_block_encoding_rewrites.py`.
