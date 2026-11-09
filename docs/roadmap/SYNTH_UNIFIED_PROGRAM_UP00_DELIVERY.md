# UP-00: unified program contract and #1003 adjustment

The [contract](../api-changes/FQ-SYNTH-UNIFIED-PROGRAM-UP00-20261102.md) fixes
meaning and compatibility before adding the program language. It preserves Qmod
concepts and unchanged supported sources. It does not implement UP-01 or export
new role/type objects. Path: single_device_fast_path, exact local CPU only.

## Delivered boundary

- One Compiler-owned program semantics, using existing HybridProgram and CoreIR;
  no new runtime, semantic engine, public builder or contract record class.
- Separate pure expression, reusable quantum function and prepared root;
  initialization, roles, fresh assignment/XOR and clean ownership are explicit.
- Input means consumption; Const preserves basis values while allowing phase
  and entanglement. Neither is a classical scalar parameter.
- Explicit phase/global-phase and control/inverse/within-apply requirements;
  upcoming coherent control keeps condition and body references disjoint.
- Current accepted subset versus each follow-on UP gate is documented. Existing
  predicate-phase extension is preserved without claiming upstream equivalence.
- The unmerged float-return native phase evaluator/flag and Services dispatch
  are removed. Public signatures, exports, snapshots and integer/pi semantics
  are unchanged. RZ/PHASE angle validation, Qmod admission and original pins stay.

## Runnable current user workflow

Save this in a `.py` file, or run `python -m examples.synthesis.tutorials.qmod.native_and_qmod_rz.main`:

```python
import flagquantum as fq
from flagquantum import synth


def energy(x: int, y: int) -> int:
    return x*x+y


program = synth.synthesize(
    energy,
    inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
    input_state=synth.InputState.uniform(),
    max_qubits=24,
    memory_limit_bytes=2**30,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y", "result"))))
```

All sixteen `(x,y,x*x+y)` outcomes have probability approximately 1/16. The
native default result name is `result`; default reading is that marginal only.
The current equivalent Qmod, also exercised in the example and replacement test:

```qmod
qfunc energy(const x:qnum<2>, const y:qnum<2>, output z:qnum) {
    z=x*x+y;
}
qfunc main(output x:qnum<2>, output y:qnum<2>, output result:qnum) {
    allocate(x); allocate(y);
    hadamard_transform(x); hadamard_transform(y);
    energy(x,y,result);
}
```

Use `synth.synthesize(source, source_format="qmod")`, then the same run/read
consumer. Tests also replace the plain Python function with an existing qfunc
snapshot and compare all sixteen complete complex amplitudes and zero workspace.
They do not require raw statevectors from different schedules to have matching
physical shapes. Result widths infer from the full input domain, not one sample.

The separate RZ migration/reference comparison is:

```python
rotation = synth.synthesize(
    "qfunc main(output q:qbit) {allocate(q);H(q);RZ(1.9,q);}",
    source_format="qmod",
    memory_limit_bytes=1024**2,
)
migrated_state = fq.run(rotation).statevector()
reference = fq.Circuit(1).h(0).rz(0, theta=1.9)
reference_state = fq.run(reference).statevector()
```

The runnable example uses complex128 for the independent Circuit and checks
both against exp(-0.95i)/sqrt(2), exp(0.95i)/sqrt(2), retaining global phase.
The H-containing source is a coherent variant. The pinned original RZ.qmod
contains no H and is still separately executed unchanged at its original size.
Circuit is labeled a gate-level reference; general native phase is a gap.

Measured current example: native arithmetic uses 18 allocated/used qubits,
10 workspace, 166 operations (4 H, 106 CX, 56 CCX), depth 124. Its CPU
complex128 planned state buffer is 4 MiB. The H/RZ variant uses one qubit,
zero workspace, two operations, depth two and 32 planned state bytes. These are
actual plan resources, not process memory, speed, optimality or hardware claims.

## Unified target example: design only, not executable today

This is the follow-on UP-01 through UP-04 acceptance target. Imports and bodies
are not runtime admission in UP-00; do not present this block as the runnable
user example above.

```python
import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, Const, QNum, allocate, hadamard_transform, phase, control


@synth.qfunc
def cost(x: Const[QNum[2, False, 0]], y: Const[QNum[2, False, 0]]) -> None:
    phase(x*x+y, 0.8)


@synth.qfunc
def kick(y: Const[QNum[2, False, 0]]) -> None:
    phase(y, 0.3)


@synth.qfunc
def main(x: Output[QNum[2, False, 0]], y: Output[QNum[2, False, 0]]) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    cost(x, y)
    control(x == 1, lambda: kick(y))


program = synth.synthesize(main, max_qubits=24, memory_limit_bytes=2**30)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y"))))
```

Qmod design counterpart, not part of the 216 original-source coverage:

```qmod
qfunc cost(const x:qnum<2,False,0>,const y:qnum<2,False,0>) {
    phase(x*x+y,0.8);
}
qfunc kick(const y:qnum<2,False,0>) { phase(y,0.3); }
qfunc main(output x:qnum<2,False,0>,output y:qnum<2,False,0>) {
    allocate(x);allocate(y);hadamard_transform(x);hadamard_transform(y);
    cost(x,y);
    control(x==1) { kick(y); }
}
```

The expected amplitude is exp(i*(0.8*(x*x+y)+0.3*I(x==1)*y))/4 for every
x,y in 0..3, with clean scratch. The control body references y, not x. This
formula is a future independent oracle, not execution evidence. Budgets shown
are candidate acceptance ceilings, not measured resources for this target.

## Verification and gaps

Final focused conformance: 163 passed in 3.86 seconds. The suite covers existing source safety, annotation/effect refusal,
pi compatibility, frontend replacement, roles, local cleanup, RZ/PHASE absolute
phases and source pins, exact numeric limits, actual budgets and target refusal.
Related regression: 2688 passed in 645.27 seconds across 82 files selected
from the retained Compiler/Ecosystem/synthesis/API regression, with the withdrawn
profile removed and the new conformance files added. Reproduce the related regression with:

```bash
python -m pytest tests/api_contract/test_synthesis*.py \
  tests/integration/test_*synthesis*.py tests/integration/test_construction*.py \
  tests/integration/test_qmod_parser_execution.py \
  tests/team/compiler/test_construction*.py \
  tests/team/compiler/test_state_preparation_synthesis.py \
  tests/team/ecosystem/test_qmod*.py \
  tests/team/services/test_synthesis_plan_checks.py \
  --ignore=tests/integration/test_synthesis_source_typing.py -q
```

The two source-overload typing fixtures are outside this 82-file selection;
strict mypy separately passes all 879 production/example sources. Public API
snapshot, architecture, dependency, generated-doc, maturity, language, local-link,
Black/Ruff and whitespace checks pass without changing their rules.

### Required default tier and baseline comparison

The full default command, `python -m pytest -m "smoke or unit" -q`, records
17417 passed, 527 skipped, 27 failed, 6 errors and 6023 deselected in 837.80
seconds. It was not stopped at the earlier three-failure limit and is not green.
Log: `/private/tmp/fq-up00-default.log`.

All 33 failure/error nodes were then selected verbatim on an isolated archive of
main base `6df1f2ae6296334d087fdf39918f8b1e46559a19`, with its own Git metadata,
read-only object alternates and a local ref snapshot to preserve the provenance
tests' history context. The same dependencies and sandbox reproduce the identical
33-node failure set: 27 failed, 6 errors in 12.11 seconds. All nine implicated
test files are byte-identical to the base. An independent numerical-only comparison
also reproduces the same nine Simulation failures on base and current source.

Failure groups are existing Simulation fusion/dispatch/batch-initialization
assertions; installed Aer entry-point discovery; local evidence-history checks;
and denied local rendezvous sockets/Gloo initialization. No affected Simulator,
backend/plugin, evidence-validator or network implementation is changed by UP-00.
These are not UP-00 acceptance passes or release evidence, and this local-context
comparison is not a claim about a fresh CI runner. CI must validate the revised
PR independently. No test assertion, skip rule, snapshot or source is weakened.

The first baseline selector truncated one parameterized node containing spaces,
so pytest ran no tests. That invalid selection is preserved separately and not
counted as baseline evidence. The corrected selector requires 33 collected nodes
before recording the comparison. Logs:
`/private/tmp/fq-up00-baseline-default-failures.log`,
`/private/tmp/fq-up00-baseline-comparison.json`, and
`/private/tmp/fq-up00-baseline-invalid-selection.log`.

Historical prototype evidence remains in
[SYNTH_LITERAL_RZ_DELIVERY.md](SYNTH_LITERAL_RZ_DELIVERY.md), explicitly scoped
to head 4c923bf111ace9c29bf0d784bbfabfc0fee940d9. Tests of the withdrawn behavior
are replaced by rejection/compatibility tests; their old passing counts are not
attributed to the revised implementation. No frozen snapshot is changed.

Next: UP-01 implements the first typed preparation/arithmetic/call/output path,
then UP-02 explicit phase, UP-03 cross-call inverse/cleanup and UP-04 coherent
control. General Input/free/drop, arrays, views, full Qmod library, runtime
high-level real parameters, hardware and release admission remain gaps. Trials
are 0/2 and release_authorized=false. No merge is performed by this delivery.
