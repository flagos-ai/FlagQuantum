# Shared extension of a singly controlled component

G2-CTRL04 supplies the compiler dependency for adding controls to an existing
controlled operation. It follows unchanged Qualtran v0.7.0 ControlledViaAnd and
specialized_ctrl, commit bcb17bb25360ed68ce3bc084645db3e929799389. Existing
controlled_component_ir and typed predicate cost rules remain authoritative.
Source copyright and links are retained in the adapted modules.

The IR function extend_controlled_component_ir receives structured new control
groups, the existing (wire, activation), a positive-controlled body callback,
clean work and classical slots. Bit encoding is the existing big-endian,
row-major ControlRegister encoding. New controls precede the old control.

For one new bit, the source computes And(new activation, old activation), calls
the positive body and erases that And. It must not use a global negative-control
X pair instead of the original negative-polarity And decomposition. For several
new bits, an outer CtrlSpecAnd reduces them, then an inner And joins that
predicate with the existing control. Both layers stay live while the body runs.
The helper reserves one work wire/result slot per new bit, with outer storage
first and the existing-control conjunction last. The body owns its private
storage, preserves parent controls/predicates, and supplies its own direction.
No primitive measurement/reset sequence is mechanically inverted.

The existing indexed specialization now consumes this generic helper when an
existing control is present. The original solitary-negative ApplyLthBloq defect
without an existing control remains on its previous explicit path. It is not
silently repaired by this extraction. Existing conjoined repeated-control roots
also remain distinct. This is algorithm-independent predicate composition, not
a second indexed traversal implementation.

controlled_component_resource_model owns concrete ComponentCost composition.
controlled_component_cost models an arbitrary structured wrapper around an
already-positive-controlled child. extended_controlled_component_cost adds the
old-control conjunction, and retains a separate outer level for multi-bit new
controls. Both reuse control_predicate_cost and its existing QEC authority.

Callers explicitly supply target_width, the child's logical THRU ports excluding
its positive predicate. It must not include the child's private peak. This
matters when an indexed child declares traversal scratch in ComponentCost.qubits:
inferring logical ports from that value would keep scratch live outside its
scope and overestimate width. Predicate calls retain payload bystanders; child
calls retain the original controls and other live predicate storage. Each metric
uses the paired child direction/profile. Models neither emit nor execute IR.

The new model tests compare 16 structured-wrapper and 32 existing-control
resource cases against original QEC categories and QubitCount, including both
QEC profiles, shaped control arrays, signed/ones-complement and stored integer
words. Eight complete complex-state tests enumerate every small predicate
measurement history and verify all work clears. Invalid storage and model
contracts fail before a body is invoked. Existing indexed source gate/model,
measurement-branch and repeated-control tests exercise the refactored consumer.

Two LCU source-only probes verify that a later returned root with three control
bits decomposes and has valid model/width, while its returned builder callback
still fails at the original malformed np.concatenate call. This repeats the
known builder defect on the actual LinearCombination path without editing the
reference or claiming native execution repairs it.

A further LCU source probe distinguishes adjoint placement. For a negative
additional control, extending a positive-controlled LCU and then taking its
adjoint uses four X operations on the added control; taking the LCU adjoint
first and then extending uses ControlledViaAnd with two X operations. Their
complete original matrices agree. Primitive export retains allocation scopes
for And, PREPARE and SELECT; flattening all scopes before Cirq export can reuse
a still-live predicate. A future LCU consumer must preserve which original
operation chain was requested rather than merging the two gate sequences.
This batch delivers the shared compiler layer and source audit, not that future
consumer or whole-file completion.

The existing user workflow remains runnable:

```sh
python -m examples.algorithms.indexed_component_controls
```

It uses circuit.append(op), adds a two-bit word to an existing zero control,
and verifies all complex amplitudes and clean work. The nested predicates and
traversal emit/model 12 T/T-dagger, three measurements and eight peak wires.
The --stacked example keeps its separate repeated-single-control root path.

Use the existing report entry point without constructing or simulating a circuit:

```python
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.control_predicate import ControlRegister
from flagquantum.compiler.registers.controlled_component_resource_model import controlled_component_cost
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

positive = ComponentCost("CX", 2, GateCost(clifford=1))
model = controlled_component_cost(positive, (ControlRegister(2, (1,)),), target_width=1)
report = estimate_ftqc_resources(model)
assert report.logical.gates.total_t_count() == 4
assert report.logical.gates.measurement == 1
assert report.emitted is None
```

An optional `emitted_ir=circuit.to_ir()` adds actual instruction counts to
the report without replacing the analytical model. QEC categories and their
T-equivalent conversion remain distinct from a fully synthesized gate stream.
Final focused validation: 70 new tests, 468 total passed without skips; strict
types, formatting, lint, ownership and architecture boundaries passed.
