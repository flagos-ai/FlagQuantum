# Block encoding rewrites

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Frozen source | Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0294 | `rewrites.py` | `209c5aad537c9cbd8458d26294bdb6eb09a5a43d` | Verified source oracle reconstruction and explicit term identity. Automatic equivalence for arbitrary callable and foreign terms remains open. |
| Q-0295 | `rewrites_test.py` | `79422a95c589b573eea88b689686508b69cd9ab1` | Closed. Both original test scenarios have native equivalents. |

The rewrite operates on unbound Algorithms block objects. It traverses
Phase, Product, and TensorProduct and accumulates nested default LCU terms
in first occurrence order. A nested term's coefficient is multiplied by
its parent's coefficient before accumulation. The output precision is the
maximum of the parent precision and the parent-plus-child value for each
nested LCU. Cancellation preserves zero-valued terms. A single remaining
unique term retains the source constructor's minimum-arity refusal.

Native term equality and hash govern the ordinary rewrite. A caller can
supply `term_key` for independently constructed interchangeable terms. The
key covers the complete term, including parameters and directional bindings.
The first representative is retained. A key of `None` uses the existing
equality, so a caller can declare equivalence only for terms it knows.

Supplied native oracles are opaque by default. The `rebuild_oracles=True`
path follows the locked rule and reconstructs every visited LCU with its
default PREPARE and SELECT. The locked constructor also clears the source's
stored control flag. Native LCU controls are supplied at binding, so
the equivalent result is appended without an outer control. Callers rebind
using the rewritten port sizes, work ownership and new term order.

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

## Reconstructed oracle cases

Four supplied-oracle configurations compare the rebuilt signature,
coefficients, error and normalization. The PREPARE-only case grows the
resource word from two to four. Nesting that block and an independently
constructed X merges the two X terms, preserves the first X binding and
produces coefficients `(1, -4)` at precision three. The symbolic case
checks source precision propagation under an enclosing Phase and
retains a zero coefficient on cancellation.

Both directions of the standalone and nested cases compare literal
primitive operands and measurement feedback to the locked exports. The
standalone case checks all four measurement histories. The nested case
checks all-zero, all-one and both alternating histories across ten
measurements. Every checked history compares the complete output for the
specified coherent input and clears private work. The independent models
agree in both resource profiles.

| Rewritten fixture | Physical emission | Modern source model |
| --- | --- | --- |
| Standalone, precision one | 22 T/T-dagger, 53 CX, 2 measurements, 8 reserved wires | 2 Fredkins, 2 And, 48 Cliffords, 2 rotations, 2 measurements, peak 8 |
| Nested, precision three | 54 T/T-dagger, 175 CX, 10 measurements, 18 reserved wires | 2 Fredkins, 10 And, 142 Cliffords, 2 rotations, 10 measurements, peak 18 |

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
