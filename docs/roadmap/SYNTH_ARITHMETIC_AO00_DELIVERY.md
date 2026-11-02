# AO-00: reproducible arithmetic synthesis baseline

AO-00 fixes the expression corpus, independent oracles and cost interpretation
before optimizing lowering. Production code and public contracts are unchanged.
The eighth planning revision's AO-01 through AO-05 remain future work.

## User workflow

Save this in a source-retaining Python file, or run
`python -m examples.synthesis_arithmetic_baseline`:

```python
import flagquantum as fq
from flagquantum import synth


def energy(x: int, y: int) -> int:
    return x*x+y


program = synth.synthesize(
    energy,
    inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
    input_state=synth.InputState.uniform(),
    memory_limit_bytes=64 * 1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y", "result"))))
print(program.report["resources"])
```

Expected: all sixteen `(x,y,x*x+y)` outcomes have probability approximately
1/16; the complete-domain result has four bits, with all auxiliary wires zero.
The user declares unsigned domains and preparation, with no gate design,
physical wire assignment, carry management or bitstring decoding.

The equivalent existing Qmod path is:

```qmod
qfunc main(output x:qnum<2>,output y:qnum<2>,output result:qnum) {
    allocate(x);allocate(y);
    hadamard_transform(x);hadamard_transform(y);
    result=x*x+y;
}
```

Pass that string to `synth.synthesize(source, source_format="qmod")` and use the
same run/read workflow. The measured AO-00 corpus uses the existing pure-function
convenience form and was originally based on main commit
`ef95c04f08ddeb05c1c8469b51f4cfccb7c4fc62`. UP-01 has since merged as #1004;
the AO-00 branch now includes that mainline implementation. The original
measurement and artifact remain unchanged. The new Python qfunc preparation and
composition workflow is demonstrated in
[the program example](../../examples/synthesis_python_program.py). UP and AO
share existing Compiler semantics and execution; this baseline does not
introduce an alternate engine. See the synchronization receipt below.

The independent two-bit Circuit reference is:

```python
import torch

reference = fq.Circuit(9, dtype=torch.complex128)
for wire in range(4):
    reference.h(wire)
reference.cx(1, 7).cx(3, 7)
reference.cx(2, 6).ccx(1, 3, 6)
reference.x(1).ccx(0, 1, 5).x(1)
reference.ccx(1, 2, 8).ccx(8, 3, 5).ccx(1, 2, 8)
reference.ccx(0, 1, 4)
```

It is derived independently for this finite domain, not called by production
synthesis. For x=2a+b and y=2c+d, low-to-high output bits are b XOR d, c XOR bd,
a(1-b) XOR bcd and ab; scratch bc is uncomputed. The runner checks the full
uniform state against mathematics; tests check all 256 input/target combinations.

## Resource meaning and observed sentinel

| Implementation/profile | Allocated/used | Workspace | Operations | Depth | Two-qubit operations |
| --- | --- | --- | --- | --- | --- |
| Python/Qmod logical | 18/18 | 10 | 166 | 124 | 106 |
| Manual logical | 9/9 | 1 | 15 | 11 | 3 |
| Python/Qmod exact named basis | 18/18 | 10 | 860 | 567 | 410 |
| Manual exact named basis | 9/9 | 1 | 92 | 60 | 39 |
| Python/Qmod named basis + line | 18/18 | 10 | 2162 | 1088 | 1712 |
| Manual named basis + line | 9/9 | 1 | 196 | 112 | 143 |

These are static counts of generated circuits, not CPU runtime measurements.
Logical CCX is one three-qubit operation, so its cost becomes visible in the
decomposed target rows. Every target row uses the same existing exact
phase-capable basis and normal target compilation. Converting fixed logical
circuits is distinct from rerunning target-aware candidate selection. The
matrix's other expressions only claim logical resource evidence.

The 18-qubit planned complex128 state buffer is 4 MiB, versus 8 KiB at nine
qubits. Neither is process peak memory. The manual result is a reference, not a
lower bound or optimality proof. No AO optimization is implemented here.

The equivalent helper-expanded `square(x)+y` retains 18 qubits/166 operations/
depth 124. Exchanging addition operands to `y+x*x` currently produces 20 qubits,
12 workspace, 170 operations and depth 126, with a 16 MiB planned buffer. AO-00
records this real normalization gap; it does not modify lowering or assert that
all equivalent spellings already have equal cost. This is a regression target
for AO-01 and later candidate selection.

## Reproduction and scope

The [benchmark guide](../../benchmarks/synthesis/README.md) specifies all ninety
cases, schema, oracles, budgets, timing boundaries and commands. The checked
[baseline JSON](../../benchmarks/synthesis/baselines/ao00.json) records each case,
including completion status and verification scope. The runner source hashes
and measured code revision distinguish source identity from the later evidence
commit. Host-dependent durations are not exact-equality assertions.

Small cases have exact complete complex-amplitude checks, not just marginal
probabilities. Larger cases have independent exact gate interpretation on an
explicit finite sample, not a whole-domain quantum or formal equivalence claim.
A timed-out case is incomplete and cannot claim a cost or successful verification.
Default source/function naming and spelling differences are not normalized away
from this initial record.

This is local CPU static-resource/correctness evidence. It makes no performance,
optimality, QPU, distributed, independent-user-trial, release or increased
original-Qmod-corpus-coverage claim. No original source/license pin is changed.

## Validation receipts

Measured code: `6fc0ad1f87b037dcbe1935d4546de5e1dbe841e6`. The artifact's six
Python-file hashes match the current measured scripts. The later evidence/docs
commit does not replace the measured revision.

Two complete runs each record 89 compiled cases and one incomplete timeout:
`mixed_products-16x16` (`x*x+x*y`). That case exceeded the fixed 30-second child
wall budget; no resource, equivalence, infeasibility or successful execution
claim is attached. Each run has 6018 independent basis/XOR/cleanup checks across
compiled cases. Thirty-seven cases execute both uniform and relative-phase
full states, hence 74 full-state checks; other cases explicitly have zero
coherent checks. The target/reference and spelling sentinels run in addition.

The two-run comparison reports `{"changed_resources_or_status": []}`: every
resource record, circuit identity, target/reference row, spelling variant and
completion status matches. Source revision and all measured script hashes also
match. Host timings are recorded, not compared as a speed claim.

- Checked baseline SHA256:
  `c5f084ae48066f8055a94a3be02d0d30d71c233783c7fd1234ccb2b3d464c864`.
- Independent repeat SHA256:
  `744932be0e85e7f76a619571ea17cc9ae37a3a376f6dd2e03e33b1a5f79ef744`.
- Repeat raw result: `/private/tmp/fq-ao00-repeat.json`; comparison command and
  checked baseline are sufficient to regenerate new comparisons. The repeat
  hash identifies this run, not a promise of identical timing JSON on another host.

Focused acceptance: 53 passed in 7.21 seconds (20 mathematical/oracle cases,
16 integration cases and 17 artifact-contract cases). It includes all 256
sentinel input/target combinations, whole complex states and clean workspace,
large-static no-execution guards, helper/operand-order cases, three-way target
costs, forced child timeout, and refusal of altered domains, Boolean widths,
overstated verification, corrupted histograms/memory and incomplete cost claims.
The user example runs successfully and prints all sixteen named outcomes.

Retained integration files pass in separate fresh processes: multiplication
12/12, existing user workflow 26/26, checked execution 17/17. An initial combined
91-case invocation had 75 passes and 16 failures due to the existing user-workflow
file leaving a patched lazy-root `fq.plan` visible to the subsequent execution
file. The unmodified main archive reproduces the same 16 failures/27 passes for
those two files in that order. No old fixture, assertion or production API was
changed to hide this ordering issue; the combined run is not claimed green.

Strict mypy passes all 879 production-package/new-example sources and the seven
benchmark/new-example sources. Black 24.1 and 26.10.1, Ruff, architecture,
dependency policy, unchanged public API guard, capability maturity, generated
document consistency, English-language policy, local links, hygiene and
whitespace checks pass. This is not a whole-historical-examples typing claim.

### Broader checks and baseline limitations

The full default smoke/unit run records 17437 passed, 527 skipped, 27 failed,
six errors and 6037 deselected in 870.92 seconds. This is not a green default
run. Its 33 failure/error node identities exactly match the previously captured
unmodified `ef95c04f...` main reproduction and UP-01's default run. The implicated
production/test files are unchanged. Existing failures include local installed
optional-package discovery, Simulation fusion/dispatch/batch-initialization
expectations, provenance context, and sandbox-denied local sockets.

The benchmark-contract/release-gate tier records 976 passed, 91 skipped and one
failure in 91.55 seconds. The failure is the existing MPS release-gate subprocess
that deliberately removes PYTHONPATH and loads an older installed FlagQuantum
from the shared environment, missing `_FORBIDDEN_FALLBACKS`. The same node fails
on the unmodified main archive with the same ImportError; no environment or
package installation was changed to bypass it.

The CPU distributed tier first encounters sandbox loopback restrictions and is
interrupted after one failure/10 passes/54 skips. Re-running with local loopback
access records 317 passed, 170 skipped and one failure in 141.33 seconds. That
existing abrupt-rank-loss test requires a peer-loss diagnostic string absent
from the macOS elastic-launcher output despite the expected injected SIGKILL and
ChildFailedError. The identical test fails on the unmodified main archive in
1.24 seconds. This baseline does not modify distributed code, relax diagnostics
or claim those tiers are green or scalability is established.

Local logs: `/private/tmp/fq-ao00-{baseline,repeat,comparison,focused,default}.log`,
`fq-ao00-retained-{multiplication,user-workflow,execution}.log`,
`fq-ao00-main-order-baseline.log`, `fq-ao00-benchmark-tier.log`,
`fq-ao00-main-benchmark-baseline.log`, `fq-ao00-distributed-tier-unsandboxed.log`,
`fq-ao00-main-distributed-baseline.log`, `fq-ao00-production-mypy.log`, and
`fq-ao00-default-comparison.json`. These are local receipts, not portable artifact
links. PR CI must independently check the final head in its installed runner
context. No AO-01 optimization, UP-01 merge, release promotion or original-Qmod
coverage increase is inferred from the baseline.


## Main synchronization after UP-01 merge

At the user's request, #1004 merged normally at its fully checked head
`eca15c0eb30e1749610d453ffeecbd64ed53401e` (57 successes, three configured skips),
producing main commit `8bb4de1b78b47664cad90270a839b35e10e730d9`. AO-00 merged
that latest main without conflict in
`1fc10e0c5f2b6641b41814adab9a45d215bf82da`. This preserves both histories;
there is no rebase or replacement measurement commit.

The original `ao00.json` bytes and SHA256
`c5f084ae48066f8055a94a3be02d0d30d71c233783c7fd1234ccb2b3d464c864` are unchanged.
Its measured source revision remains
`6fc0ad1f87b037dcbe1935d4546de5e1dbe841e6`, which is still an ancestor of this
branch. The merged main commit is also an ancestor. No raw resource artifact,
timeout status or historical receipt has been regenerated after synchronization.

The synchronized tree passes 158 combined AO-00/UP-01/retained-multiplication
acceptance cases in 19.94 seconds. These include the baseline reader, independent
oracles, both Python entry forms, Qmod, coherent states, arbitrary-target XOR,
clean workspaces, source/role/alias/budget refusal and the current resource
sentinels. This targeted post-sync run is distinct from the full pre-sync
baseline and default-tier receipts above. Final-head PR CI independently
checks the complete merged tree.

Post-sync strict mypy passes 882 production/two-example sources. Both
`examples.synthesis_arithmetic_baseline` and `examples.synthesis_python_program`
run successfully: the former preserves its original named-outcome/cost workflow,
and the latter compares Python in-program preparation with Qmod on all sixteen
outcomes. The public API guard and all 810 local Markdown links pass after the
merge. Logs: `/private/tmp/fq-ao00-up01-sync-focused.log`,
`fq-ao00-sync-mypy.log`, `fq-ao00-sync-example.log` and
`fq-ao00-sync-program-example.log`.
