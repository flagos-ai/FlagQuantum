# SABRE decay experiment: not admitted to the default compiler

Status: rejected development candidate on 2026-11-21. Compiler owns this
`single_device_fast_path` investigation. Production compiler code, public APIs,
routing options, defaults, schemas, dependencies and capability maturity remain
unchanged. The experiment patch is archived for reproduction only.

## Question and upstream mechanism

Would one additional deterministic decay-scored SABRE trial improve delivered
application circuits enough to justify more planning work? The control is
FlagQuantum main `cf5fc36c29c3f9162b6b58054f8163c98944e33d`, including its
existing five trials: deterministic, seeds 0/1/2, and Cirq-derived layer lookahead.
The experiment keeps those opportunities and adds one candidate; it does not
remove an existing trial or change a competitor.

The mechanism was inspected in Qiskit 2.6.2, commit
`0c26a5a158c4d62b2b303a8c2f3c6eb2b14653b4`,
[`route.rs`](https://github.com/Qiskit/qiskit/blob/0c26a5a158c4d62b2b303a8c2f3c6eb2b14653b4/crates/transpiler/src/passes/sabre/route.rs)
Git blob `90847d4032770bddf2d1c699595a985e20962317`:
`choose_best_swap` multiplies the absolute distance score by the maximum decay
factor of the SWAP endpoints. `swap_map_trial` increments selected endpoints,
periodically resets decay, and resets factors after executing ready operations.
The installed Python SABRE pass uses increment 0.001 and reset interval 5.
The existing IBM Apache-2.0 attribution in the native planner is retained by the
patch. This is an adaptation of that scoring mechanism, not an implementation
of the complete Qiskit router. Existing Cirq-derived mechanisms are unchanged.

The prototype multiplies the existing absolute frontier/lookahead score by
`max(decay[left], decay[right])`. Multiplying only a relative distance delta
would be incorrect because different candidates have different multipliers.
For simplicity this experiment uses full absolute scoring even above 12 qubits;
it does not establish the performance of an optimized relative-score version.
Endpoint factors increase by 0.001 after a heuristic SWAP, reset every fifth
search step, and reset after executing ready operations. The step counter
survives the latter reset. Forced recovery SWAPs do not update decay. The decay
trial does not enable paired layer-lookahead SWAPs.

A candidate is selected only if its forward plus identity-restoration SWAP count
is strictly lower than the incumbent. The existing directed-cost callback is
preserved by the application hook, but all measured devices here are undirected.
Automatic strategy selection, native legalization, target block synthesis and
artifact construction remain in the application workflows. In particular,
selecting a cheaper internal route need not change the delivered output.

## Results

All inputs use public, previously inspected development generators. No task was
drawn prospectively after configuration freeze.

| Suite | Cases | Quality changes | Semantic checks |
| --- | ---: | --- | --- |
| Generic CX routing | 80 | 4 lower SWAP counts; 76 ties; 1 logical-depth regression | Complete exact GF(2) operators and every target edge |
| Application workflows | 120 paired cases | 1 changed delivered circuit; 119 equal CZ/depth pairs | Both versions pass the unchanged SDKBench evaluator |

Generic inputs cover widths 6/8/12/16, line and two-row grid topologies, seeds
0–9 and `4 * width` CX operations. Total delivered SWAPs fall from 6694 to 6670:
24 fewer, approximately 0.36%. The depth regression is width 16, line, seed 7:
SWAPs 282 to 272, but logical two-qubit dependency depth 109 to 117. Logical CX
and SWAP each count as one in this suite; this is not physical CZ depth.

Application inputs cover QAOA, Ising, fermionic hopping, QFT and coherent
repetition recovery; widths 6/8/12; line/ring/grid/sparse targets; and ordinary
restored compilation plus the explicit target-artifact workflow. The fixed
public nonce is `01` repeated 32 times. SDKBench is unchanged at
`9aa9260920984e2ca05bcb293abb885fccd9c144`. Six-qubit checks compare the complete
operator up to a global phase; widths 8 and 12 check three complete output probe
states and do not establish universal operator equivalence. Both sides restore
identity output layout, use no ancillas, and meet the existing native target.

The only changed delivered case is ordinary six-qubit QFT on a line: physical
CZ count 117 to 111, but two-body dependency depth 74 to 79. The corresponding
target-artifact workflow retains the same delivered CZ count and depth. Across
both versions of all application cases, maximum reported semantic error is
`6.072657351995283e-14`, below the unchanged `1e-8` threshold.

Timing is diagnostic only: one unwarmed shared-host observation per case, with
no independent memory enforcement or exclusive-host qualification. Generic
baseline planning totals about 0.375 seconds and the additional trials about
0.073 seconds in the archived reproduction. These are sums of planning
observations, not whole-compiler timings, confidence intervals or speedup claims.
They illustrate extra work; they cannot establish a stable latency penalty.
Application wall times include evaluation and process startup and are not
compiler performance measurements.

## Decision and limits

Do not add this sixth trial to the default compiler. The measured delivered
benefit is narrow, includes a depth regression, and requires extra work in a
compiler whose latency remains an improvement target. This rejects this specific
adaptation and admission policy, not decay scoring in general or Qiskit's
implementation. A better scoring implementation, trial replacement, or a
separately justified count/depth policy would need new evidence.

The next routing investigation should identify why restoration and route
selection leave delivered CZ/depth gaps, before increasing the search budget.
It should measure complete outputs and compile work together. It must retain
neutral references, all failures and current layout/deployment constraints.
No SDK ranking, superiority, hardware fidelity, memory or scalability claim is
made. Formal SDKBench host qualification remains deferred. No host configuration
change or restart is needed for this experiment.

## Reproduction

Use disposable FlagQuantum and SDKBench checkouts at the commits above, Python
3.12 and the existing development dependencies. The archived environment used
Torch 2.14.0. Copy the benchmark module and patch from this change into the
FlagQuantum checkout if reproducing directly from the base commit. From the
FlagQuantum root:

```bash
git apply --check docs/development/evidence/sabre-decay.patch
git apply docs/development/evidence/sabre-decay.patch
python -m benchmarks.compiler_sabre_decay --suite generic \
  --output /tmp/sabre-decay-generic.json
python -m benchmarks.compiler_sabre_decay --suite applications \
  --sdkbench-root /path/to/unchanged/sdkbench \
  --output /tmp/sabre-decay-applications.json
python -m pytest tests/unit/test_compilation_sabre_trials.py \
  tests/unit/test_compilation_sabre_layout.py \
  tests/unit/test_compilation_sabre_progress.py \
  tests/integration/test_compiler_application_cz.py -q
git apply --reverse docs/development/evidence/sabre-decay.patch
```

The runner refuses an unpatched compiler or a changed/wrong-revision SDKBench
checkout. Each paired application case has a 150-second process watchdog with
process-group cleanup; this is a development watchdog, not the formal SDKBench
resource protocol. Results are checkpointed after every case, including failures.
The script enables one thread per numerical library in each child. The control
always runs before the candidate, so the timestamps cannot support paired timing
claims. The 80 generic metric rows reproduced exactly against the initial probe;
all 120 application cases were also rerun through the committed runner.
The focused compiler checks passed 72 tests with the prototype applied.

Raw observations: [generic](evidence/sabre-decay-generic.json),
[applications](evidence/sabre-decay-applications.json),
[archived patch](evidence/sabre-decay.patch), and
[runner](../../benchmarks/compiler_sabre_decay.py). Successful stdout JSON is
removed only when it equals the retained parsed result; numerical evidence is
preserved. The runner deliberately requires an explicit patch so this rejected
prototype cannot be selected by an ordinary compiler invocation.
