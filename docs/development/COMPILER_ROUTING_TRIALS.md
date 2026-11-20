# Bounded native routing trials

Status: integrated in PR #1233 at `2d2390767e1697bf0630f0be86fb4b6d07fabad0`. Compiler
owns this single-device CPU planning change. No public signature, routing option,
IR schema, dependency, or final-layout contract is added or changed.

## Upstream mechanism and native adaptation

Qiskit 2.6.2 revision `0c26a5a158c4d62b2b303a8c2f3c6eb2b14653b4`
[`State::choose_best_swap` and `swap_map`](https://github.com/Qiskit/qiskit/blob/0c26a5a158c4d62b2b303a8c2f3c6eb2b14653b4/crates/transpiler/src/passes/sabre/route.rs)
select among equal-cost SWAP candidates using a seeded generator, run multiple
routing trials, and select a minimum-SWAP result. The existing IBM Apache-2.0
notice in the native planner covers this adaptation. Cirq-derived progress
recovery remains in place; no additional Cirq mechanism is claimed here.

FlagQuantum now retains its original deterministic route and evaluates three
additional serial trials using local random generators with seeds 0, 1 and 2.
They shuffle candidate order before the existing strict score comparison. The
process-global random state is untouched. This is fixed-budget diversity, not
Qiskit's complete decay, scheduling, layout or parallel trial implementation.

The selection objective is the delivered route: forward SWAPs plus the existing
final restoration. On directed topologies it also includes both SWAPs of every
required operand-orientation repair. A new candidate replaces the original only
when this cost is strictly lower. Equal-cost results keep the original.
The parent automatic selector still compares its existing deployable strategies;
calibrated devices still refuse the uncalibrated SABRE planner.

The planner remains behind the same IR/plan boundary. Tests replace only trial
selection to recover the original implementation, without modifying callers,
and compare complete delivered operators and costs. No upstream SDK object
crosses into the compiler and no external SDK is invoked during compilation.

## Layout and search bounds

Using the best new forward route to generate the next backward layout can erase
an older good layout: development regression cases exposed this directly. Layout
candidate generation therefore retains the original deterministic forward and
backward trajectory. Each candidate is separately evaluated using bounded trials.
Identity placement stays in the candidate set. This preserves the old layout
opportunities while permitting cheaper routes for each candidate. Nonidentity
expert layouts retain the existing restoration and deployment limitations.

Each routing call evaluates at most four trials, and a route with no forward
SWAPs returns immediately. Existing five-round layout bounds remain unchanged.
An additional trial stops when its committed forward SWAP count cannot improve
on the incumbent delivered cost. Only SWAPs before the last executed instruction
are committed: progress recovery can undo the later speculative suffix, so that
suffix must not trigger premature pruning. The established frontier, lookahead,
cycle recovery, and topology checks remain authoritative.

Extra trials execute sequentially; they do not multiply simultaneous planner
state by four. They do add planning work. This implementation offers a count
bound, not a wall-clock or process-memory guarantee. Caller time/memory limits
remain necessary. Materialization reuses the retained plan and initial layout
instead of running layout search again.

## Development results and limits

The reproducible probe uses 80 ordinary generated CX circuits: widths 6, 8, 12
and 16, line and two-row grid topologies, seeds 0 through 9, and four times the
width in source instructions. These inputs were inspected during development;
they are not prospective SDKBench tasks. The baseline differs only by disabling
trial selection. Both sides retain progress recovery, relative scoring and final
restoration.

| Width | Topology | Original SWAPs | Trial-selected SWAPs | Improved cases / 10 |
| --- | --- | ---: | ---: | ---: |
| 6 | Line | 250 | 248 | 1 |
| 6 | Grid | 162 | 154 | 3 |
| 8 | Line | 530 | 518 | 5 |
| 8 | Grid | 294 | 286 | 2 |
| 12 | Line | 1262 | 1252 | 4 |
| 12 | Grid | 662 | 634 | 6 |
| 16 | Line | 2466 | 2448 | 5 |
| 16 | Grid | 1262 | 1228 | 7 |

Across all cases, SWAPs decrease from 6888 to 6768 (120, approximately 1.74%).
Thirty-three cases improve and 47 tie. All emitted routes pass exact complete
GF(2) operator comparison and coupling-edge checks. This reference covers the
complete CX/SWAP operation, not only its all-zero output. It does not prove
phase-sensitive correctness for other gate families; separate state, symbolic
parameter and gradient tests cover those workflows.

Logical two-qubit dependency depth improves in 27 cases and worsens in two.
The probe counts a logical SWAP as one operation; this is not native CZ depth.
Fewer SWAPs alone cannot establish shorter duration or improved fidelity.
Timing rows contain one sample on a shared host without warmup, so they are
retained as diagnostics and cannot establish a speedup. Extra trials generally
cost more planning time. No memory, hardware or cross-SDK result is claimed.

Raw rows, source fingerprint and machine details are retained in
[evidence](evidence/sabre-routing-trials-development.json). Reproduce with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.compiler_sabre_trials > /tmp/sabre-trials.json
python -m pytest tests/unit/test_compilation_sabre_trials.py \
  tests/unit/test_compilation_sabre_layout.py \
  tests/unit/test_compilation_sabre_progress.py -q
```

Historical exact single-trial count assertions are retained using the original
selector replacement. Improved outputs must not exceed those counts. The moved
layout scenario now uses a case that still selects a nonidentity placement;
identity can legitimately win after route improvement. A separate test proves
layout search runs once when materializing a retained plan.

## Existing user workflow

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

source = fq.Circuit(8).h(0).cx(0, 7).ry(3, theta=0.2).cx(3, 6)
compiled = compile(source, coupling_map=CouplingMap.grid(2, 4))
assert compiled.metadata["routing"]["mapping_restored"] is True
```

The existing automatic path now prices bounded SABRE trials alongside its other
strategies. Its output retains the existing topology, provenance and restoration
contracts. Independent SDKBench candidate freezing, resource-qualified runs,
matched layout controls, full-CZ/depth/time/memory reporting and replay are still
required before drawing comparative conclusions. Bounded paired-SWAP lookahead
is integrated separately in [PR #1234](COMPILER_LAYER_LOOKAHEAD.md); broader
initial-placement search remains open work.

## Validation record (2026-11-19)

The final compiler selection passed 2,976 tests across unit compiler suites,
compiler team conformance and application/artifact integration paths. This
includes thirty new trial-selection, pruning, random-state isolation, directed
cost, complete-operator and plan-reuse cases. The vocabulary census passed all
60 cases. Whole-package mypy plus the new probe passed 1,133 source files on
Python 3.12. Scoped Ruff, Black formatting, English-language checks, documentation
links, compiler ownership and the protected public API snapshot passed.

These checks do not replace the complete smoke/unit and runtime CI tiers. The
branch includes the relative-scoring and progress-recovery prerequisites; its
full review and integration depend on those changes passing their own gates.

The complete local smoke/unit run at `338f9bb50` finished with 20,734 passed,
786 skipped, 15 failed and six fixture errors. Eight architecture failures
identified this change exceeding the existing routing-module size ceiling;
reusing the coupling-map reference removed repeated property expressions and
restored the ceiling without changing it. The 84 affected architecture,
interop-boundary and routing checks then passed. Six launch-test failures and
six Gloo fixture errors came from denied loopback access; all 103 tests in those
two files passed with the required local socket access. The remaining evidence
revision test reported 98 differences. The checker, declarations and evidence
files are unchanged from the main baseline in this worktree; this is still an
unresolved local qualification failure, not a waived gate. The original full
run remains failed. Subsequent PR CI permitted integration; the candidate and
current qualification scope are recorded in [the upstream audit](COMPILER_UPSTREAM_REUSE.md).

## Historical lookahead experiment and production trials

The W7-04 LookaheadSwap experiment compared a beam search to the deterministic
single-trial planner. Its smallest depth/width anchor cannot reproduce a
multi-trial selection by definition. The live benchmark now measures explicit
`sabre/single-trial` and `layout/single-trial` controls alongside production
`sabre` and `sabre_layout`. Reference layout trials also retain the single-trial
control, so the historical comparison does not silently change its algorithm.

The original 1.03 ratio bound remains against its original deterministic layout
control. Tests additionally require current production planned totals to be no
worse than their deterministic controls. All configurations retain numerical
and topology checks. Historical JSON is unchanged; current ratios to production
are still reported and may differ when production improves. This repairs the
comparison identity rather than raising a threshold or forcing production to
emit an older, more expensive route.

The earlier 98 local evidence-origin errors were checked in isolated complete
object stores containing only the selected published branch ref. Both main
`45468b934a18e552a2a69dcc0b8f1db0d8fa16f9` and the synchronized candidate
`a8901e23635a753972d68ddb1ab59cb9b35fd7d8` passed the unmodified evidence-origin
checker. The producing workspace retains additional historical refs that change
the checker's reachability answer. No origin declaration or checker was changed.
The earlier full smoke/unit run remains a failed run, not a passing certificate.
