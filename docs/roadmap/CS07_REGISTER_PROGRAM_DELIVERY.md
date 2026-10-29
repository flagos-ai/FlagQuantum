# CS07-B fresh register initialization and composition

Date: 2026-10-30. Baseline: merged #965,
`320b623592cbd26ace328b2bae3439463e899fa6`. Its exact head
`e7f4b5971611db0881b8ad9e86533fad13e89947` completed 57 successful checks and
three configured skips before merge. This stage implements the Compiler-owned
part of the [shared CS07 contract](../api-changes/FQ-SYNTH-CS07-20261029.md).
Classification: single_device_fast_path, private development evidence.

## User problem and the current public path

Imported Qmod needs fresh outputs, ordered quantum preparation and multiple live
outputs. An immutable arithmetic expression alone cannot describe those effects.
The public Python workflow already implemented on main remains:

```python
import flagquantum as fq
from flagquantum import synth

def polynomial(x: int, y: int) -> int:
    return x * x + y

program = synth.synthesize(
    polynomial,
    inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
    input_state=synth.InputState.basis(x=3, y=2),
    max_qubits=18,
)
assert dict(program.probabilities(fq.run(program))) == {(11,): 1.0}
```

Save the definition in a Python source file. Run
`python -m examples.algorithms.synthesis_trial` for the maintained basis, uniform
and routed examples. This stage does not ask users to construct internal records,
write gates, assign physical ports or decode bitstrings. The new representation
is a private adapter/compiler seam, not an additional end-user language or API.

## What changed and what was reused

| Component | Implementation and reason |
| --- | --- |
| Register reference | Existing IRType gains a private quantum.register(width) form; a live storage reference cannot be confused with a quantum.uint expression or classical INDEX constant. No new model class or public IR schema |
| Register program | Existing HybridProgram/Region/Block/Value/Operation and SourceLocation, with one existing linear QUANTUM_EFFECT argument. The new verifier tracks initialization and the current effect; it does not replace the pure-expression verifier |
| Fresh initialization and gates | Fixed-width declarations, zero allocation, fitting unsigned literal initialization, indexed X/H/S/Z. Numeric index i maps to MSB-first ports[w-1-i] within that register |
| Arithmetic composition | Existing verifier, lifetimes, pure kernels and retain/recompute schedule. Each call borrows initialized registers, computes a fresh output and cleans workspace before later effects |
| Repeated actual references | An ordinary qint.call wrapper deduplicates entry borrows and retains repeated operands inside the expression graph; existing binding validation remains strict and unchanged |
| Composite evidence | One private RegisterCandidate holds the complete circuit, named registers and actual ArithmeticCandidate segments. A program with multiple outputs and ordered mutation has no single immutable expression lifetime graph |
| Candidate selection and target checks | Existing selection code now factors its common consumer for the two concrete builders. Target lowering, final resource checks, deterministic ranking, recovery, stage identity and incomplete-search refusal have one implementation |

No Core/Runtime/Simulation/Services implementation, public export, stable contract,
serialization or dependency was changed. The existing arithmetic row's circuit
still excludes external preparation, preserving its current report and execution
binding. Register rows contain their whole initialization/preparation/cleanup.
The private CandidateRow retains its existing field name `arithmetic`, now able
to hold either the original ArithmeticCandidate or the composite RegisterCandidate;
common cost/selection consumers use their circuit and peak_workspace fields.

## Exact profile and lifecycle obligations

A register program has one ordered block with only an effect argument and no
borrowed external quantum inputs. Declarations have fixed positive unsigned
widths and unique nonempty names. They begin uninitialized. Allocation or literal
initialization must occur exactly once; gates and arithmetic cannot read an
uninitialized object. Fresh arithmetic assignment refuses an already initialized
destination, input/destination aliasing, mismatched input widths and truncation.

Effects consume the current token and produce exactly one successor. Undefined
or mistyped values, reused definitions, stale effects and nested regions fail
before emission. All declared outputs must be returned initialized, once and in
order. There is no implicit output drop, storage reuse or release. Input/Const,
free/drop/bind, general arrays/structs, signed/fractional types, dynamic control,
measurement/noise and arbitrary imported gates are outside the closed profile.

Output references can be read repeatedly. Arithmetic borrows them only for the
current segment; its computed temporaries are inverse-cleaned immediately. Later
mutation therefore cannot invalidate a deferred inverse. The named output pool
is reserved throughout; only the tail proven clean by an arithmetic segment is
reused for another segment. Peak workspace is the maximum actual segment peak,
not the sum or a count inferred from final CoreIR.

The emitted fresh-register circuit requires all-zero storage at entry. An
allocation/effect declaration is not evidence that an arbitrary existing state
is zero. Future Services admission must bind the whole fresh preparation and
its real execution through the existing checked-plan authority. Compiler does
not run, inspect or authenticate a state. ConstructionBudget still bounds source
construction/search independently from final resource ceilings.

## Evidence and limits

The new suite has 62 passing cases: 50 pure verifier/selection cases and 12 real
CPU integration cases. It covers fresh qbit X, coherent two-bit square-plus-one,
multiple quantum constant outputs beside a product, repeated actual aliases,
asymmetric numeric bit weights, two sequential computations separated by input
mutation, and width/depth schedules with real SWAP/nonidentity line layout.
Full complex128 statevectors match independent analytic references within 1e-12;
actual mapped dirty-workspace probability is at most 1e-12. Simulated cases stay
at or below 22 qubits.

For the one-bit pressure model with H/H/S preparation, retain has six workspace
qubits, ten total qubits and 43 source operations; recompute has five workspace,
nine total qubits and 61 source operations. A final nine-qubit ceiling selects
recompute; a 43-operation ceiling selects retain. The same limit checks include
all preparation, target transformations and cleanup. Rejected complete requests
have real checked retries; incomplete construction has no best-so-far success or
invented impossibility claim.

The original tutorial's declared x width three and inferred expression output
width six are constructed and checked structurally. This is model evidence,
not native parsing or original-source execution. The original corpus remains
source-integrity 3/3, parsed/lowered/executed/full-state verified 0/3. Two-bit
probes never replace those declarations or increase compatibility counts.

Coverage of the two new modules: 175 statements and 92 branches, 100% in this
focused suite; this is not whole-package coverage. The first coverage attempt
segfaulted in Torch native import before collection; preloading Torch before
starting coverage allowed the entire unchanged suite to run. No test was removed
or skipped. Whole-package strict mypy passes 870 files.

Final expanded regression passed 552 cases in 43.02 seconds, covering all
construction stages, existing selection, Services plan/readout, public admission
and the real subprocess example. The default smoke/unit run reached 4,398
passes and 121 skips before the same three existing native-CPU QFT fusion-count
failures stopped it (187.08 seconds; 4,375 deselected; --maxfail=3). Those macOS
cases observe 44 operations rather than 28 without the native CPU extension.
No Simulation code or expected counts were changed; full local defaults are not
reported as green.

Changed-file Ruff/Black, API migration, architecture, dependency, team scope,
vocabulary, repository language, local documentation links and generated-doc
checks passed. No stable contract/snapshot or existing numerical expectation
was weakened to admit the new profile.

## Measured static costs and the original-size execution boundary

These costs come from the actual shared emitter and existing target estimator,
with the default logical target. They are static model evidence; no native Qmod
was parsed and the larger two models were not simulated.

| Model and schedule | Total qubits | Peak workspace | Source operations | Final operations | Final depth |
| --- | --- | --- | --- | --- | --- |
| One-bit pressure, retain | 10 | 6 | 43 | 42 | 36 |
| One-bit pressure, recompute | 9 | 5 | 61 | 60 | 55 |
| Three-bit square-plus-one, either schedule | 23 | 14 | 379 | 379 | 288 |
| Three-bit a=4/b=5 with six-bit product, either schedule | 24 | 12 | 279 | 275 | 214 |

The last row is a conservative fixed-width model, not an upstream serialized
width oracle. Inferring the source's omitted qnum representations remains the
adapter's obligation. The original-size equivalent models exceed the routine
22-qubit full-state test grid. CS07-D must explicitly preflight the memory and
execution budget for any larger full-state acceptance, or reduce workspace with
a general verified transformation. It must never shrink the original declared
x width or use a reduced probe to fill the original-file verification count.

## Next boundaries

CS07-C owns bounded native-source parsing, roles and numeric representation
adaptation into this admitted model, with explicit source/version diagnostics.
No parser, Classiq SDK object or filename dispatch is added here. CS07-D then
composes it with existing Services checked execution and reading, verifies the
three unmodified originals at declared sizes and separately records parse,
lowering, execution and full-state evidence. A new public external-source entry
still requires its own proposal. This stage adds no .fq parser or run_qmod API.

Independent developer trials against the current public Python path remain
pending. The [trial guide](SYNTH_DEVELOPER_TRIALS.md) is a handoff, not two human
records; release authorization stays false. Private Compiler correctness does
not certify external compatibility, hardware behavior, optimality or usability.
