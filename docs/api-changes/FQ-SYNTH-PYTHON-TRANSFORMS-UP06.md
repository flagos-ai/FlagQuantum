# Native Trigonometric Transforms

UP-06 adds native Python qfunc calls for the bounded cosine/sine transforms
already present in the checked register IR and lowering:

```python
from flagquantum.synth import QNum, Output, allocate, qct_type2, qfunc

@qfunc
def main(state: Output[QNum[3]]) -> None:
    allocate(3, state)
    qct_type2(state)
```

The public operations are `qct_type2`, `qst_type2`, `qct_qst_type2`, and
`qct_qst_type1`. They require a complete, initialized, writable target of one to
four qubits. The combined Type-II operation also requires a disjoint initialized
one-bit selector: zero selects cosine, one selects negative sine. Type I keeps
the positive imaginary phase on its sine sector. Partial target views and
selectors aliasing the target are rejected.

The Python frontend emits the existing `qreg.type1_transform` and
`qreg.type2_transform` records. Register verification, bounded lowering, resource
accounting, and instruction scheduling remain shared; there is no new executor,
serialized operation, or duplicated transform decomposition. Implementation is
authorized for the UP-06 native corpus only. Release remains unauthorized.
