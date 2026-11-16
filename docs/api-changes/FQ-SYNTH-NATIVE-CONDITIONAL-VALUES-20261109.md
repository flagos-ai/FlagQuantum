# Native unsigned conditional values

Status: unreleased extension to the captured Python expression subset.
Classification: `single_device_fast_path`; `release_authorized=false`.

Typed qfunc fresh assignment accepts `yes if predicate else no`. The predicate
is a one-bit quantum value, a comparison, or a literal zero/one. Both branches
must be valid unsigned expressions, even when the predicate is constant. Width
inference retains enough bits for either branch. Nested conditional values and
checked quantum helpers compose under the existing ownership rules.

This is coherent value selection: it neither measures the predicate nor runs a
Python branch on a quantum value. The existing pure Python capture produces
`qint.select`; register computation, lifetime analysis, workspace scheduling,
uncomputation and budget checks retain their existing authority. No operation,
public export, schema, simulation kernel or Qmod adapter is added.

```python
from typing import Annotated

import flagquantum as fq
from flagquantum.synth import Output, QNum, allocate, hadamard_transform, qfunc, synthesize


@qfunc
def main(a: Output[QNum], b: Output[Annotated[QNum, 3]], res: Output[QNum]) -> None:
    a |= 4
    allocate(b)
    hadamard_transform(b)
    res |= a if a < b else b


program = synthesize(main)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("a", "b", "res"))))
```

The unchanged minimum original and native program are compared with an
independent reversible truth-table Circuit. Generic tests cover minimum,
maximum, ties, nested and constant predicates, wider branches, complex coherent
states under both objectives, absolute phase, clean workspace, helper composition,
alias rejection and budgets. See
[`test_synthesis_native_selection.py`](../../tests/integration/test_synthesis_native_selection.py).

Host calls, dynamic classical effects, signed/fractional values and multi-bit
numeric conditions remain refused. Python `min`/`max` calls are not captured by
this extension. The two retained `max(3.5, a)` originals require fractional
semantics and remain pending; an integer conditional is not a substitute.
