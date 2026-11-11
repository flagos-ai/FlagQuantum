# Classical truth table inspection

Simulation owns `classical_truth_table.py`: scalar domain enumeration and text
formatting adapted from the unchanged Qualtran v0.7.0 `get_classical_truth_table`
and `format_classical_truth_table` at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Source attribution remains in the
native file. Six original Git blobs match the frozen inventory, including the
classical simulator, its tests, data types, registers, composite graph and Add
implementation. Workspace evidence is `g2-classical-truth-table-source.json`.

## Ownership and interfaces

`classical_truth_table(signature, domain, evaluate)` consumes the existing Core
`Signature`, one numerical domain provider per input register and a bound action
returning output values in signature order. It returns input headings, output
headings and a materialized list of input/output sequence pairs. It adds no Core
type, graph, registry, dtype dispatcher or dependency. Existing bit, bounded,
fixed-point, ones' complement and Montgomery domain owners remain authoritative.
The action can call the existing `ComponentClassicalState` on Compiler-supplied
call positions; Simulation does not construct or transform that graph.

The complete input-shape scan happens before any domain query. Any shaped input,
including a zero-length shape, raises the source's `NotImplementedError`.
Domain providers are queried in input signature order; `itertools.product`
materializes pools before the first action. Duplicate values, original scalar
types and order remain intact. An empty input signature evaluates once; an empty
domain suppresses evaluation. Right-only shapes are not refused by this helper.
Output sequences retain object identity, even when their arity differs from the
headings; action/domain exceptions propagate without translation or rollback.

`format_classical_truth_table` keeps the original double-space headings,
separator length including the heading newline, comma-separated values and
arrow rows. It uses each value's formatting protocol rather than normalizing
arrays, numbers or symbolic values. It adds no escaping or output-arity policy.

These are internal numerical inspection helpers on the `single_device_fast_path`.
They do not emit instructions, select a backend or certify physical execution.
Word enumeration is separate from physical register bit order; append emitters
retain their existing MSB-first bindings and clean-work ownership. The normal
quantum user workflow remains `circuit.append(op)`.

A small numerical consumer reuses the existing CNOT value owner:

```python
from collections.abc import Sequence

from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.bit_encoding import bit_value_domain
from flagquantum.simulation.classical_controlled_bits import cnot_classical_values
from flagquantum.simulation.classical_truth_table import (
    classical_truth_table, format_classical_truth_table,
)


def evaluate(values: dict[str, object]) -> Sequence[object]:
    control, target = values["control"], values["target"]
    assert isinstance(control, int) and isinstance(target, int)
    return cnot_classical_values(control, target)


table = classical_truth_table(
    Signature.build(control=1, target=1),
    lambda register: bit_value_domain(),
    evaluate,
)
assert len(table[2]) == 4
print(format_classical_truth_table(*table))
```

## Verification and physical consumer

- 38 new tests compare the two unchanged source functions, including nine dtype
  domains, repeated values, NumPy scalar identity, empty domains, directional
  ports, same-named opposite sides, shaped outputs, arbitrary output sequences,
  provider failure traces and exact text including embedded newlines.
- A real two-Add(2,3) native component graph is evaluated through
  `ComponentClassicalState` for all 64 inputs including a spectator. Its full
  truth table equals the original composite graph's table. The reconstructed
  64-by-64 complex matrix matches the original tensor contraction.
- That matrix acts on a normalized complex superposition; all 16 unnormalized
  measurement branches of the existing emitted circuit match it with factor
  1/4, and both work bits return to zero. Concentrated regression also retains
  the original gate/operand/feedback comparisons and inverse histories.
- Concentrated regression: 2,056 passed, 207 recorded reference/native numeric
  warnings. New tests alone: 38 passed, three original Autoray matmul warnings.
  The new owner covers all 26 statements and eight branches (100%).
- CPU/representation/tensor-network regression: 84 passed, two known optional
  dependency/deprecation warnings. Strict typing of the two new files, Black,
  Ruff, Simulation scope, architecture, multi-level IR and protected API pass.

- Default smoke/unit run: 7 failed, 21,890 passed, 300 skipped, 1,037 warnings in
  120.30 seconds, stopped at its maxfail bound. All seven failure IDs equal the
  recorded baseline; no new failure was observed. This is not a fully green suite.

## Separate resource accounts and remaining boundaries

The enumeration/formatting owner emits zero primitive gates and owns zero
physical work bits or classical feedback slots. It materializes domain pools
and all table rows: storage grows with the product of input domain cardinalities.
There is no source QEC cost model for the inspection helper.

The existing two-Add consumer emits H12, T8, Tdg8, CX40, S4, measurement4,
conditional CZ4 and reset4, with two clean work bits and four feedback slots.
The unchanged source QEC model is AND4, Clifford22 and measurement4. These
categories are recorded separately from primitive counts; numerical enumeration
does not change the physical circuit or its model. No GPU, hardware or
scalability result is claimed.

Truth tables describe classical values, not quantum phases or branch
probabilities. The caller must supply the correct value/phase policy, e.g. the
pure native graph evaluator rejects any explicit phase contribution rather than
silently certifying them. Input arrays remain unsupported as in the source.
Finite-field/dtype graph traversal, foreign lazy decomposition exceptions and
complete original test-file replacement remain open. The helpers can enumerate
existing standalone dtype domains without implying executable graph admission
for those encodings. Q-0882 and Q-0883 remain open, and no new child ID or parent
denominator is introduced. G3 remains unstarted.
