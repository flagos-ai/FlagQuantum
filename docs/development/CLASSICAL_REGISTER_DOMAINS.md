# Classical register domains

Simulation owns `register_classical_domain(Register)` and
`register_classical_truth_table(signature, evaluate)`. Core retains neutral
metadata and executable admission; Compiler retains physical lowering. No new
public export or executable encoding is admitted.

The domain adapter selects the existing bit, bounded integer, fixed point,
ones-complement, Montgomery and finite-field numerical owners. Signed and
unsigned integer domains follow the pinned `_Int/_UInt.get_domain` methods.
Concrete width/parameter extraction is shared with the existing codecs, replacing
their private duplicate. The field owner reconstructs the full field projection,
including coefficient/polynomial identity and classical storage.

```python
from flagquantum.core.ir.program.register_types import unsigned_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_truth_table import register_classical_truth_table

signature = Signature((Register("word", unsigned_register_type(2)),))
table = register_classical_truth_table(signature, lambda values: (values["word"],))
```

The table owner preserves input order, Cartesian ordering, output objects and
formatting. It checks all input shapes before querying domains. The existing
`classical_truth_table(signature, domain, evaluate)` remains the explicit callback
entry for custom domains and symbolic metadata. Opaque domains raise TypeError;
unknown encodings fail closed. Error text uses native context. The adapter accepts
concrete metadata only; its domain acquisition can reject metadata eagerly, while
source `get_classical_domain` usually defers iteration through a generator.

Bounds are not repaired to binary capacity. Fixed point enumerates stored
integers. Montgomery with a concrete modulus omits zero and can yield an empty
domain. Odd fields enumerate typed field elements rather than every binary word.
A zero-width signed domain retains the source negative-shift failure. Dense
materialization costs the full Cartesian product. No phase or stochastic policy
is inferred from a truth table.

## Verification and resources

`tests/unit/test_classical_domains.py` compares 17 word declarations and 16 field
projections directly with unchanged Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. It checks original domain methods and
truth-table/formatting functions, zero widths, empty bounds/moduli, typed field
values, shape-first refusal, custom callbacks and unsupported metadata.

A replacement test connects the new domain adapter to the existing native Add
graph/table consumer without changing physical lowering. All 64 classical inputs,
the complete matrix and all 16 measurement branches preserve complex amplitudes;
both work bits return to zero. The physical instructions are CX 40, H 12, T 8,
Tdg 8, S 4, measurement 4, CZ 4 and reset 4. The independent source QEC model
reports AND 4, Clifford 22 and measurement 4 (16 T from four ANDs). These are
different counting vocabularies, not equal primitive totals. Domain enumeration
itself emits zero gates and uses no quantum work bits.

Source-wide dtype execution, foreign decomposition exceptions, shaped classical
ports and original dense field type loss remain outside this slice. The reference
source is unchanged. G2 remains open; this does not start G3.

Batch checks: 53 new tests, 3556 concentrated regressions and 84 CPU/TN checks passed. Strict typing passed for six changed Python files. Domain dispatch (30 statements/16 branches), shared metadata (13/8), truth table (29/8) and codec (51/28) reached 100%; the broader measured owners reached 97% collectively. Seven pinned source blobs matched the inventory.

Default smoke/unit: `7 failed, 22234 passed, 300 skipped, 1037 warnings in 245.15s (0:04:05)`. Observed failures belong to the seven known baseline IDs; 0 baseline IDs were not observed before the bounded stop. This is not a green full-suite result. Evidence: `ftqc-reference-env/g2-classical-domains-default-comparison.json`.
