# Import measured external circuits

The optional Ecosystem module `cirq/_feedback_import.py` converts an explicitly
ordered binary Cirq circuit to owned CircuitIR. It is the feedback-import
dependency of Q-0299's foreign sparse-oracle binding. The existing static
`import_cirq` contract and stable exports remain unchanged.

```python
import cirq
from flagquantum.ecosystem.cirq._feedback_import import import_cirq_feedback

control, target = cirq.NamedQubit.range(2, prefix="q")
source = cirq.Circuit(
    cirq.measure(control, key="m"),
    cirq.X(target).with_classical_controls("m"),
    cirq.reset(control),
)
converted = import_cirq_feedback(source, qubit_order=(target, control))
assert converted.report.lossless
assert converted.ir.instructions[1].metadata["conditions"] == ((0, 1),)
```

The caller chooses wire ordering, including idle wires. Named Qids are accepted;
duplicate, missing and nonbinary wires are rejected. No external SDK objects
enter the returned IR. The module loads Cirq only when conversion is requested.

Each measured bit gets a native classical slot. Repeated keys retain every
record, and KeyCondition supports the source's historical record index.
Multibit keys have the source big-endian integer semantics. Invert masks change
the recorded bits with native classical XOR; they do not flip the quantum
post-measurement state. `interop.measurement_records` maps serialized key names
to tuples of native slot words, separately from temporary predicate slots.

Supported Sympy guards include Boolean And, Or, Not, Xor, Implies, Equivalent
and ITE, equality and disequality of measured words with integer constants or other words,
and indexed recorded bits. Conditions lower compositionally to the existing
native classical XOR and conditional-assignment spelling. They do not enumerate
an exponential truth table or allocate quantum predicate work. Out-of-domain
integer equalities retain their exact constant truth value. Unknown keys,
unavailable historical records and out-of-range bit indices fail explicitly.
Key symbols use canonical assumptions, matching the source resolver's
name-based substitutions; symbols with additional assumptions are rejected.

Reset after measurement and conditional reset preserve source semantics.
The latter uses the existing guarded-effect representation. Global phase uses
the shared exact scalar-phase compiler primitive on a declared anchor wire,
including when guarded by a classical predicate. Ordinary gates reuse the
static importer's admitted primitive lowering. This importer performs no gate
synthesis or reference-specific dispatch.

Unsupported gates, tags, noise/confusion maps, conditional measurements,
unsupported symbolic arithmetic or ordering comparisons, and scalar phase without an anchor raise a
conversion error with a diagnostic report. There is no lossy mode. A dependency
object requiring decomposition must first expose its actual admitted primitive
operation tree at this optional boundary.

The tests compare the original pinned And erasure and external gate matrices
and predicate resolution against every native raw measurement branch. They
cover repeated records, result inversion, multibit words, conditional reset,
spectator wires, Boolean expressions and both guarded and unguarded scalar
phases. In the pinned And example the imported primitive stream is exactly H,
measurement, conditional CZ and reset, with one result slot and no extra
quantum wires or predicate assignments. Larger conditions expose their actual
native classical assignments; these are separate from the source's quantum
gate model.

An independently defined foreign square-root entry oracle is also decomposed
with the pinned source builder, including its temporary And target. Its
imported primitive stream matches the existing native callback on every wire,
including the measurement result and feedback, and every branch matches the
external instrument. This establishes conversion before physical relocation.

The complete snapshot oracle entry now consumes this dependency; see
[Circuit sparse oracles](CIRCUIT_SPARSE_ORACLES.md) for forward/adjoint binding,
quantum/classical relocation and explicit workspace/output ownership.

```bash
pytest -q tests/unit/test_cirq_feedback_import.py
```
