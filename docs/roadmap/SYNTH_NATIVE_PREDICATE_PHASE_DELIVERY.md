# Direct native predicate phases

Follow #994, merged at ea1d2da8d34ecd35ccf4b68a7222408d9f2b03af after
57 successful checks and three configured skips. See the
[contract](../api-changes/FQ-SYNTH-NATIVE-PREDICATE-PHASE-20261101.md).

The [Classiq phase documentation](https://docs.classiq.io/qmod-reference/language-reference/statements/phase/)
specifies polynomial expressions. Direct unsigned comparisons are explicitly a
FlagQuantum-native extension of the import profile; they do not increase upstream
original-source compatibility evidence. Source-version context remains declared,
not authentication. The four-original cohort and source/license bytes are intact.

## Complete user workflow

```python
import flagquantum as fq
from flagquantum import synth

source = """qfunc main(output x:qnum<2>,output y:qnum<2>) {
    allocate(x); allocate(y); hadamard_transform(x); hadamard_transform(y);
    phase(x<y,pi);
}"""
program = synth.synthesize(source, source_format="qmod", max_qubits=9)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y"))))
```

Run `python -m examples.synthesis.tutorials.control.native_predicate_phase.main`. Each pair has
probability 1/16; full amplitudes are (-1)**int(x<y)/4. Users need no temporary
flag declaration, assignment or explicit scope. Existing Python predicates remain
available through the same Program/run/reader workflow.

## Reuse and semantics

Only adapter.py changes in production: remove the polynomial-only rejection and
its redundant comparison-tracking flag. Existing expression inference, relation
normalization, qreg.phase, Compiler predicate kernels, selected cleanup, resource
selection, exact-phase target guard and checked execution remain authoritative.
No new public API, IR, operation, allocator, role checker, executor or dependency.

Exactly-one-bit quantum-dependent expressions may include the six unsigned
relations, including relations nested in admitted arithmetic. Width inference
uses complete declared domains, not initialized basis values. Literal-only
phases, wider results, arbitrary coefficients, chains/logical operators,
uninitialized values and unsupported target/memory conditions still fail closed.
Explicit locals remain useful when several actions reuse a computed value.

On CPU complex128 / Torch 2.13.0, direct x<y uses nine allocated/used qubits,
five hidden work bits, 49 operations (4 H, 14 X, 16 CX, 14 CCX, 1 Z), depth 40.
The equivalent explicit-local producer requires ten qubits. The direct producer
fits max_qubits=9 while the scoped form is rejected. These are measured costs,
not a global optimality, speed or hardware claim.

## Verification and limitations

198 new cases pass: six relations over sixteen input pairs under two objectives,
complex reference entanglement with coherent control in inline/Const-helper
forms, replacement of an explicit local and an interference test. Existing source
refusals retain unsupported literal-only or multibit cases where comparisons
are now positive. Original digests remain unchanged.

The first broader run had 821 passes and one failure: an old negative still
expected phase(x==1,pi) to fail. The negative now retains the literal-only
phase(1==1,pi) refusal; the new interference test positively exercises q==1.
The initial failed run is not recorded as green. Whole-package mypy 2.x passes
879 production/example sources, and the public API baseline passes unchanged.
Final broader results and current CI status are recorded in the PR.
The corrected broader run passes all 822 cases in 65.77 seconds, covering
318/318 statements and 144/144 branch exits in the single production module.
The first failed run is retained separately; overlapping counts are not added.

The prior local default run (on #994, before this extension) completed with
5027 passed, 121 skipped and three pre-existing QFT fusion assertion failures
in 203.98 seconds. Those Simulation files are unchanged; no unrelated assertion
or implementation is modified here. This does not certify the entire local
default suite. Native phase target/memory and parser/refusal coverage are reused
through the existing broad source workflow tests.

No arbitrary angles, mutable/consuming roles, Input/free/drop, arrays/views,
general control/inversion, SDK/cloud or hardware admission. Independent trials
remain 0/2 and release_authorized=false.
