# AO-01: pure arithmetic normalization, sharing and bit facts

AO-01 follows #1006, merged as `28ebf6d27c869d0f360b36b547b12557800b3a98`
after its exact head passed 57 checks with three configured skips. It implements
the eighth planning revision's pure unsigned value stage. No public API,
production dependency, canonical IR schema, selector budget/default, executor,
release status or original-Qmod coverage is added or changed.

## Unified user workflow

Run `python -m examples.synthesis.tutorials.arithmetic.normalization.main`, or save this in a `.py` file:

```python
import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Const, Output, QNum, allocate, hadamard_transform


@synth.qfunc
def energy(x: Const[QNum], y: Const[QNum], result: Output[QNum]) -> None:
    result |= (x+y)*(y+x)


@synth.qfunc
def main(x: Output[QNum], y: Output[QNum], result: Output[QNum]) -> None:
    allocate(1, x)
    allocate(1, y)
    hadamard_transform(x)
    hadamard_transform(y)
    energy(x, y, result)


program = synth.synthesize(main, memory_limit_bytes=64*1024**2)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y", "result"))))
```

Expected: `(0,0,0)`, `(0,1,1)`, `(1,0,1)`, `(1,1,4)` each with probability
approximately 1/4; result storage has three bits and all workspace is zero.
The compiler proves `x+y` and `y+x` equivalent and shares their value. Users do
not name temporary registers, design arithmetic gates, uncompute values, assign
physical wires or decode bitstrings. Star import from `flagquantum.synth` also
provides the same approved public names; source files use explicit imports for
typing/linting. One-bit inputs keep this runnable complete-state example small;
the fixed corpus exercises other widths without pretending all are simulated.

The corresponding Qmod uses the same preparation and helper composition:

```qmod
qfunc energy(const x:qnum<1>,const y:qnum<1>,output result:qnum) {
    result=(x+y)*(y+x);
}
qfunc main(output x:qnum<1>,output y:qnum<1>,output result:qnum) {
    allocate(x);allocate(y);hadamard_transform(x);hadamard_transform(y);
    energy(x,y,result);
}
```

Use `synth.synthesize(source, source_format="qmod")`, then the same run/read
workflow. Both frontends enter the shared Compiler; neither adapts through a
second implementation or Python-to-Qmod text bridge.

## Independent Circuit reference

For these one-bit inputs, `(x+y)^2` has output bits `xy`, zero, `x XOR y`:

```python
import torch

reference = fq.Circuit(5, dtype=torch.complex128)
reference.h(0).h(1)
reference.cx(0, 4).cx(1, 4).ccx(0, 1, 2)
state = fq.run(reference).statevector().reshape(2, 2, 8)
```

The three output wires are 2/3/4 MSB first; wire 3 remains zero. The reference
uses no workspace. Its complete state and all 32 input/nonzero-target XOR
combinations are checked independently. It is not a production special case or
a lower-bound claim. The automatically compiled one-bit program improves from
AO-00's 17 qubits/12 workspace/99 operations/depth 82 to the observed
14 qubits/9 workspace/79 operations/depth 66. Dense complex128 state buffers
change from 2 MiB to 256 KiB, not process peak memory. This remains more costly
than the specialized five-wire reference; AO-02/03 kernels are not implemented.

## Architecture and scope

Only Compiler internals change: `arithmetic_rewrites.py`, `bit_facts.py`,
`lifetimes.py` and `lowering.py`. Existing source capture, HybridProgram,
verification, binding, retain/recompute schedules, arithmetic kernels, resource
estimation, target compilation, Program/run and named reads are reused.

The pass accepts a closed verified flat unsigned graph after pure calls inline.
It canonicalizes constants and commutative arithmetic, shares identical typed
operations, removes zero/one identities where type equality permits, and prunes
dead pure expressions. A quantum constant retains a typed witness instead of
illegally replacing a quantum return with INDEX. Arguments and declared result
widths are preserved, including deliberately oversized outputs. Input bounds
always cover the full declared domain, not the prepared basis value.

Bit facts record effective width and known zero/one masks. Addition propagates
possible carries; multiplication uses conservative generic trailing-zero/parity,
constant-power-of-two and same-value square rules. An integer square has bit one
zero; an odd square additionally has bit two zero. Equal widths/ranges alone do
not prove equal values. These facts are proof/optimization evidence, not a new
physical packing scheme or square kernel. Unsupported effects, regions, alias
and initialization violations are checked before simplification. Node/width
ceilings remain 4096/256; bit transfer is bounded by the admitted graph/width.

The normalization record retains the original SSA, replacements, source
positions, square identities and facts. Existing lifetime analysis now retains
explicit SSA call locations as well as body positions. Python scalar helpers
already inline before lifetime analysis; their supplied helper-body locations
remain available, while call sites discarded by that existing frontend are not
fabricated. No cross-`qreg.compute`, gate, region, request or global cache shares
values; the original register dependency/version verifier remains authoritative.

## Conservative resource guard and original candidates

Within each of the two existing schedules, the same emitter builds original
and rewritten pure value graphs. Use the existing estimator/allocation events
to compare `(peak workspace, operations, depth)`. Choose the rewrite only if
no metric increases. Keep the original candidate and SSA, including when the
rewrite is declined. An original construction-capacity failure can be replaced
by a verified rewritten candidate that fits; preserve the actual original
failure diagnostics instead of manufacturing a candidate. A normalization node
budget refusal keeps a valid original with its refusal evidence. Semantic or
binding errors are not swallowed or bypassed.

Optional duplicate emission has a private deterministic work bound: when a valid
original already exceeds 10,000 operations, keep that original and the normalized
SSA/facts, and record `construction.normalization.budget` without building another
circuit. This bounds extra circuit construction, not wall-clock time. It applies
to all expressions, not a named corpus case. A resource-refused original can
still be rescued by normalization under the existing construction limits. This
guard does not promise compile-time nonregression; AO-04 owns broader search.

A real accepted test example `a=x+y; b=y*x; c=y+b; d=y+a; return d*c` under
recompute has original logical cost `(28,869,659)` and normalized cost
`(25,863,667)`: fewer work qubits and gates, but greater depth. The guard therefore
keeps the original. This is a local monotone policy, not a target-aware global
optimum: it may decline a width/depth tradeoff a later selector could prefer.
AO-04 owns that target-aware multi-implementation choice. Original and normalized
costs and the retained candidate are private evidence, not new public report
fields. Existing candidate_checks=2 and incomplete-search behavior are retained.

The phase-mode emitter keeps its existing exact prefix/inverse behavior.
Constant simplification in value mode can outperform that older phase path;
this PR makes no phase-resource-improvement claim. Historical phase-versus-
general-reference tests explicitly use the unnormalized emitter for their old
reference and retain all whole-state/absolute-phase checks.

Old cost assertions are preserved against retained original candidates where
normalization legitimately changes results: zero addition and multiplication by
one can now use zero work, and redundant zero chains can fit despite the original
rematerialization-depth refusal. Tests still check the original bounded refusal,
actual optimized capacity, exact XOR and clean work; no original artifact or
semantic assertion is relaxed to hide a regression.

## Frozen baseline and reproduction

`benchmarks/synthesis/baselines/ao00.json` remains byte-for-byte unchanged:
SHA256 `c5f084ae48066f8055a94a3be02d0d30d71c233783c7fd1234ccb2b3d464c864`,
measurement commit `6fc0ad1f87b037dcbe1935d4546de5e1dbe841e6`. All six measured
benchmark source files are unchanged. New AO-01 measurements use a new artifact
and the same fixed ninety-case corpus, mathematical oracles, full-state bounds,
seed, targets and 30-second large-case wrapper:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m benchmarks.synthesis.arithmetic_baseline \
  --output /tmp/ao01.json
python -m benchmarks.synthesis.normalization \
  --baseline benchmarks/synthesis/baselines/ao00.json \
  --current /tmp/ao01.json --output /tmp/ao01-comparison.json
```

The comparison refuses an altered original SHA256 or different configuration,
and distinguishes improved/unchanged/tradeoff/regressed/incomplete. Timeouts
never count as successful measurements. The guard compares pre-target logical
costs; target conversion/optimization/routing can change tradeoffs. Actual
post-target counts are reported without a blanket target nonregression claim.
No runtime speedup, optimality, peak RSS, QPU, release, independent-user-trial or
distributed scalability is claimed.

## Measured resources and acceptance receipts

`ao01.json` records measurement source revision
`4cd4bafc0ae1fb5564afe2409a9a5d2117054aaa`; `ao01-comparison.json` compares
that artifact with the frozen original. Of ninety cases, 89 compile and one
(`mixed_products-16x16`) reaches the unchanged 30-second external deadline,
as it did in AO-00. Among comparable completed cases, 21 improve and 68 remain
unchanged; none regress or trade one reported logical metric for another.
This is a statement about this corpus/run, not a universal target guarantee.
An independent second complete generation at the same source revision matches
all ninety statuses, circuit hashes, resource counts, strategies, verification
records, nine target sentinels and two spelling variants. Timing samples are
retained separately and are not required or claimed to be identical.

| Case | AO-00 allocated/work/operations/depth | AO-01 |
| --- | --- | --- |
| `shared_square-1x1` | 17 / 12 / 99 / 82 | 14 / 9 / 79 / 66 |
| `shared_square-2x2` | 30 / 20 / 352 / 280 | 26 / 16 / 314 / 248 |
| `energy-2x2` | 18 / 10 / 166 / 124 | unchanged |
| `shared_square-16x16` | 170 / 104 / 41680 / 31724 | unchanged; optional emission capped |

The generator performs 6,018 basis/target checks and 74 bounded coherent-state
checks (37 cases). Large rows use exact gate interpretation with declared
sampling/exhaustiveness and no invented full-state execution. For example,
the two-bit shared-square row plans 1 GiB but executes no quantum state; the
one-bit runnable program uses 256 KiB. All state-based workspace leakage checks
are zero. The nine target sentinel counts and spelling variants remain recorded.

An exploratory uncapped run produced a second timeout at `shared_square-16x16`;
its repeat completed it. These observations motivated bounding optional duplicate
emission rather than hiding a failed measurement. Final generation completes
that case with its original resources; timings remain host-load-sensitive and
do not establish compiler speedup. Exploratory artifacts/logs remain local under
`/private/tmp/fq-ao01-exploratory*` and `fq-ao01-repeat*`; they do not replace
either checked baseline.

Validation receipts:

- Related compiler/construction/synth regression: 2,479 passed in 582.65 seconds
  before the final optional-emission guard. Final focused acceptance: 184 passed
  in 9.64 seconds, rerunning every changed test module and covering that guard,
  flattened-node budget refusal,
  full declared outputs, both schedules, coherent phases and nonzero targets.
- Strict mypy passes 883 production/example sources. Black 24 and 26, Ruff,
  public API protection, architecture, dependency, capability maturity, language,
  hygiene and Compiler ownership checks pass. The runnable native/Qmod example
  passes with the four expected outcomes and actual 14/9/79/66 counts.
- The earlier broad local default tier reports 17,584 passed, 527 skipped,
  27 failed and six errors. All 33 failure/error node identities also occur in
  the AO-00 and UP-01 baseline logs: optional Aer/plugin and shared installed
  package mismatches, simulation expectations, evidence pins and sandbox/socket
  prerequisites. This is not an all-green default-tier claim or a final-head
  rerun of that entire tier.
- Benchmark/release tier: 995 passed, 95 skipped, one known no-PYTHONPATH
  subprocess failure importing `_FORBIDDEN_FALLBACKS` from the older installed
  package. CPU-distributed tier with socket permission: 317 passed, 170 skipped,
  one existing macOS abrupt-rank-loss diagnostic-string failure. These broader
  runs precede the final narrow emission guard; no unrelated code/test is relaxed.

Local logs: `/private/tmp/fq-ao01-related.log`, `fq-ao01-final-acceptance.log`,
`fq-ao01-final-mypy.log`, `fq-ao01-final-checks.log`,
`fq-ao01-final-measurement.log`, `fq-ao01-final-repeat.log`,
`fq-ao01-default.log`, `fq-ao01-benchmark-tier.log` and
`fq-ao01-distributed-tier.log`. Remote PR CI remains a separate acceptance gate.

## PR #1010 CI repair

The first CI attempt at `3a6fbfb5d7274d179b316cf7ce2dac25ffdb807e`
exceeded the source-preparation job's five-minute deadline during artifact
upload. Downstream suites did not execute; their aggregate gates failed on
skipped prerequisites. The unchanged-head rerun cleared source preparation.

That rerun then exposed one actual stale resource assertion in
`test_workspace_example_runs_the_fixed_cost_table_and_both_cpu_paths`, reported
by both [runtime](https://github.com/flagos-ai/FlagQuantum/actions/runs/41405088616/job/125151671408)
and [coverage](https://github.com/flagos-ai/FlagQuantum/actions/runs/41405088616/job/125151672915).
AO-01 canonical operand ordering improves the existing workspace example:
two-bit retain becomes 25/36/733/543 (work/qubits/operations/depth), and
three-bit retain becomes 41/59/2702/2033. The executable output test now asserts
those actual costs, including the corresponding improved recompute gate counts.
The historical CS-05 table and original-candidate resource assertions remain
intact. No production code, semantic check, compiler budget, measured source or
baseline JSON changes in this repair.

The stale assertion was reproduced before editing. After repair, all 197
combined AO-01/workspace-execution acceptance cases pass in 11.74 seconds,
including both exact CPU schedules, every one-bit nonzero XOR target, complex
entanglement, controlled relative phase and compute/mark/uncompute cleanup.
The broader latest-source compiler/synth regression passes 2,482 cases in
594.44 seconds; both source-typing cases and the native/Qmod example also pass.
Logs: `/private/tmp/fq-1010-workspace-before.log`, `fq-1010-fixed-acceptance.log`,
`fq-1010-related.log`, `fq-1010-source-typing.log` and the downloaded CI failure
logs `fq-1010-runtime-failed.log` / `fq-1010-coverage1-failed.log`.
