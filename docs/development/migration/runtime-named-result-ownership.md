# Runtime named-result ownership and direct-seam retirement

This package consolidates the complete bounded named-result reader in
`runtime/results/readout.py`, native-result type admission in the existing
`runtime/results/adapters.py`, and removes three internal argument-only
relays. It does not add a numerical algorithm or a new public API. The previous
qualified baseline is PR [#1353](https://github.com/flagos-ai/FlagQuantum/pull/1353),
head `557f25c127e52a237e05aea9ad4c50196d81158f`.

That baseline passes all three workflows at attempt 1: 432 successful checks and
four existing conditional skips. All 384 CPU reports, 14,259 Runtime nodes,
38,568 coverage nodes, and 83 original scientific singletons are verified.
Whole-job maxima, including preparation/upload and excluding queue, are Runtime
497 seconds, coverage 603, unit 416, and distributed CPU 419. One logical-cat
partition exceeds the preferred 600-second target by three seconds; its original
workload is retained. Coverage aggregation passes in 129 seconds, including a
27-second data download. Its independent timeout retry is not triggered.

## One owner for each operation

| Previous internal entry | Authoritative result |
| --- | --- |
| `synth/_readout.py` | Complete reader, diagnostics and `Distribution` move together to `runtime/results/readout.py`; old file is deleted. |
| `runtime/results/mps.py::native_mps` | Existing type-admission implementation moves unchanged into `runtime/results/adapters.py`; all callers change and the old file is deleted. |
| `runtime/results/mps.py::mps_marginal` | Relay is deleted. The Runtime reader calls the existing Simulation marginal directly at the same lazy, no-grad MPS invocation stage. |
| `runtime/dynamic/_conditions.py::instruction_is_dynamic` | Relay is deleted. Current callers use Core's existing `is_classical_control_instruction`; actual condition validation stays intact. |
| `runtime/executors/statevector/planning.py::validate_distributed_statevector_plan` | Free relay is deleted. Consumers use retained public `DistributedStatevectorPlan.validate`; its local target becomes the existing trace builder, which retains actual validation and report construction. |

The numerical MPS marginal, Core predicate, actual statevector validator and trace
builder are unchanged. No old private module, forwarding function, rebinding or
`__getattr__` facade is added. Historical Services and intermediate Synth reader
paths resolve in the migration ledger directly to the same Runtime owner,
without ambiguous duplicate source identities. Native-result admission still
calls `result.native()` and checks the actual MPS type; absent native output
continues to raise rather than being silently converted to `None`.

Moving only the marginal call into Synth would let a northbound frontend import
Simulation. The complete reader therefore moves together: checked execution
association, representation admission, read-cost bounds, norm/nonnegative checks,
existing probability extraction, thresholding and read diagnostics. Runtime
assembles a view of an existing result; it does not plan, re-execute the program,
construct another Circuit, or duplicate a Simulation kernel. Existing sampling
and measurement owners remain authoritative.

The predicate remains a key-presence query, including for malformed metadata
that a later parser refuses. No validation is moved earlier. The MPS marginal
continues to return unnormalized probabilities; neither its semantics nor its
sixteen-qubit refusal bound is replaced with the distinct generic measurement
normalization/parity fallback.

## Public boundaries retained

`flagquantum.synth.Program.probabilities` keeps its exact signature, defaults,
return annotation, program validation and public namespace. It invokes the
Runtime-owned result operation. The existing signature is recorded in
`contracts/synthesis-phase-admission-proposal.json`; this move does not implement
that document's unapproved phase-mode proposal.

`DistributedStatevectorPlan.validate` keeps its method, signature and report
shape. Changing its internal call target does not remove the public operation or
the real validation. Existing root gradient conveniences, checkpoint/result
schemas, registered PassSpec/AnalysisSpec callback identities and protected
extension methods remain untouched, with their earlier bases recorded in the
[independent migration record](../INDEPENDENT_STRUCTURE_MIGRATION.md).

`Distribution` is an implementation type returned through the existing public
method, not an additional Synth namespace export. Its mapping behavior,
diagnostics and return annotation are preserved. The old private module is not
kept for serialization: the original object already refuses pickling because it
contains a mapping proxy. The original characterization records that refusal;
no new pickle locator, serializer or compatibility promise is introduced.

## Original evidence and acceptance

The original 140 helper outcomes remain unchanged: 102 returns and 38 refusals,
covering predicate key presence/malformed inputs, complex64/complex128 MPS
marginals including gradients/input graphs, and distributed-plan reports/refusals
through both the old free function and existing public method.

A separate 62-case corpus is captured before edits from qualified #1352's own
installed native wheel, SHA-256
`990ab01d82b3caddec8eb130dc59586a993d43e76a86a97b78d17a1e01570103`.
Its checked reader/admission sources are byte-identical to the qualified #1353
baseline. The original outcome digest is
`038678f141b049f20523ac05695a55571e3d21e5a05133d0626beb4a2eb72df0`:
18 returns and 44 refusals using real `synthesize`, `fq.run` and
`Program.probabilities`, without fake checked plans. Profiles include two-qubit
statevector/mixed results and the admitted 83-qubit native Python MPS workflow.
Full sealed-plan, register and actual state/native snapshots are recorded before
and after reading, including limits, identity, normalization, filtering and
native-type admission. Only the reader's observational elapsed-time value is
fixed during reading; planning, execution and existing scientific clocks are
unmodified. The earlier unadmitted wide Qmod attempt is retained as a refusal,
not converted into MPS admission or discarded as a passing experiment.

The 13-path implementation preview and 606-row ledger proposal have passed
ownership-scope review. They are preparation evidence, not qualification of the
new implementation. Acceptance still requires complete original-outcome replay,
unchanged public signatures, optional-import boundaries, architecture/API/docs
checks, complete default/focused tests, a newly packaged native wheel with old
private files absent, repository-external installation verification and this
package's own exact-head CI. No predecessor result substitutes for those checks.

Incoming #1084 remains draft and conflicting at
`7f32a36f9a87958e67dbe7eeb4c8c7a7081e150d`. Its earlier Services reader supports a
different, dense-only profile and is not an identical donor for the complete
current reader. Its old paths must be reconciled against the PR's own merge base
and the current authoritative owners, preserving current mixed/MPS capabilities.
Raw full-incoming-tree stale imports include historical files unchanged on that
branch; they are not all new changes to port or new defects. This package does
not claim integrated G0, whole-architecture closure, hardware capacity or Rust
execution.

## Existing user workflow

```python
import flagquantum as fq
from flagquantum import synth

program = synth.synthesize(
    "qfunc main(output q:qbit) { allocate(q); H(q); }",
    source_format="qmod",
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("q",))))
# {(0,): 0.5, (1,): 0.5}
```

The same program validation, exact result association and read-cost limits apply.
