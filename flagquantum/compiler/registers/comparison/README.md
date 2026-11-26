# Register comparison

This package owns equality and ordering with coherent, measured and mixed operands.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [less_than.py](less_than.py). Preserve register order, feedback slots and caller-owned
workspace. From the repository root, run the focused scenario tests.

```bash
python -m pytest tests/unit/test_register_comparison.py -q
```

The [register package guide](../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.

Stored executable graphs live in `constant_component.py`, `equality_component.py`,
`ordering_component.py`, `linear_component.py`, `controlled_component.py` and
`half_component.py`. Shared widening/cast/restore sequencing belongs to
`_linear_words.py`; sign extension belongs to the bits package. Each graph uses
MSB-first inputs and returns actual physical bindings after lowering. Half
comparisons transfer owned difference/flag outputs to their consumer, which
supplies that exact image for measured erasure. Source ge/lt inverse output
aliases and signed linear overflow semantics are preserved explicitly.

Run `python -m examples.algorithms.register_half_comparison` for the normal
`circuit.append(op)` path. Run
`python -m pytest tests/unit/test_comparison_component.py -q` for source call
connections, gates, feedback, full complex amplitudes, clean work and resources.
[The complete source audit](../../../../docs/development/COMPARISON_FILE_AUDIT.md)
records all frozen definitions and retained admission/source boundaries.

`ordering_declaration.py` owns generic Comparator and ParallelComparators
signatures and source direct models. Symbolic dimensions remain metadata until
explicit finite `program()` conversion delegates to the existing directional
graph owners. The comparison bit and network histories are retained logical
outputs, while comparison temporaries are cleaned internally. The shortest
checks are `tests/unit/test_sorting_declaration.py` and
`tests/unit/test_sorting_foreign_file.py`.

`controlled_equality_component.py` owns both scalar control polarities of Equals.
It composes controlled word XOR, the shared zero predicate and restoration;
negative activation adds literal X/positive-body/X. Predicate scratch is private,
while the control, words and dirty target remain caller-owned. Run
`python -m pytest tests/unit/test_controlled_equality_component.py -q` for the
original calls, primitive operands/feedback, all small complex columns and
complete erasure histories. Model Toffoli-to-T conversion is reported separately
from physical CCX and T-family inventories.
