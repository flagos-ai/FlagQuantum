# Selection-local SABRE dependencies

Compiler owns this `single_device_fast_path` implementation refinement. It
reduces repeated preparation inside the existing five-trial search. It does not
change the trials, seeds, scoring, pruning, recovery, layouts, restoration,
public signatures, serialization or SDKBench configuration.

## Requirement and lifetime

The deterministic incumbent builds its quantum and classical dependency graph.
Previously every additional seeded or layer-lookahead trial rebuilt that same
graph. SWAPs and physical layouts do not change source dependencies. The existing
Qiskit-style independent search trials and Cirq-style layer projections therefore
need separate remaining indegrees and placements, but can read the same successor
lists. This is an FQ-owned implementation refinement of already adapted routing;
it is not another upstream code port. Existing source notices remain in place.

`_build_sabre_dependencies` extracts the existing graph construction without
changing edge order or multiplicity. The selector builds one graph for its four
additional trials; the incumbent keeps its independent preparation, including
its original validation order. Each trial copies the initial indegree tuple
into its own list and only reads the shared successor lists. Placement, frontier,
lookahead, rollback and random state remain trial-local. Five-trial search now
builds two graphs. An already-local program still builds only its incumbent graph.

The graph lives only within one `_select_sabre_trials` call. It is not stored on
an IR, coupling map, global cache or returned plan. A later call re-reads mutable
instruction metadata, including measurement writes and classical conditions.
No identity or content-hash cache is introduced. Standalone and layout-search
trials keep their existing behavior; this change does not add cross-round reuse.

## Fixed development controls

Freeze this implementation and `benchmarks/compiler_sabre_dependencies.py`
before observing timing. Compare against integrated main
`2ea5b78b6449b837db151e0d29c90895f6b51a4e`. Reuse the existing inspected
Fourier, parity-ladder, graph-cost and matching-layer constructions at widths
6/8/12/16/24/32: 24 pairs. Use a 2-by-3 grid at width six and four-row grids
otherwise. These are constructed development controls, not unseen scientific
application tasks.

Both sides clone the trial and selector AST into equivalent module namespaces
and patch the same two internal call sites. Every call gets a fresh coupling
map. Measure complete explicit SABRE routing, materialization and routing
evidence, with one warmup and seven alternating retained samples. GC is outside
each timer. A separate full-call Python traced peak follows timing. Require
identical output IR hashes in every baseline and candidate call. Retain all
samples, including regressions, with source and driver fingerprints.

Development admission requires identical outputs, every case median time ratio
at most 1.10 and every traced peak ratio at most 1.05. A 600-second owned process
group bounds the whole experiment. Do not change inputs, limits or thresholds
after observations. These checks do not qualify process RSS, exclusive-host
latency, statistical significance, default-auto compilation, cross-SDK ranking,
hardware or scalability. Formal resource-qualified SDKBench remains deferred.

## Existing user workflow and verification

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

source = fq.Circuit(6).h(0).cx(0, 5).ry(2, theta=0.2).cx(2, 4)
compiled = compile(source, coupling_map=CouplingMap.line(6))
assert compiled.metadata["routing"]["mapping_restored"] is True
```

Expected: equivalent operations on legal device edges with restored output
placement. No new option is needed. The default strategy can use this existing
SABRE search; the timing control above isolates explicit SABRE.

```bash
python -m pytest tests/unit/test_compilation_sabre_dependencies.py \
  tests/unit/test_compilation_sabre_trials.py \
  tests/unit/test_compilation_sabre_classical_readers.py -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m benchmarks.compiler_sabre_dependencies --output /tmp/sabre-dependencies.json
```

Focused tests cover graph lifetime, unchanged individual bounded/unbounded
trials, duplicate quantum edges, all classical readers before overwrites, and
metadata changes between calls. Existing independent operator, symbolic-gradient,
dynamic execution, recovery, directed-topology and routing-conformance tests
remain required, alongside exact-head CI before integration.

Before measurement, type checking found two test/runner annotation defects. A
follow-up test incorrectly mutated the constructor input mapping, which the IR
copies, rather than the stored metadata. Its run passed 60 tests and failed the
lifetime case. The test now narrows the stored mapping to its actual dictionary
type before mutation. The compiler implementation, cases and resource thresholds
did not change. These setup failures are not passing evidence.

## Retained observations

The published pre-measurement head
`64f53dea91f485caa070ed790145ab5999b7bef0` completed all 24 pairs under
the fixed process budget. Every complete output IR hash matched in all 432
baseline/candidate calls, including warmups and separate memory calls. All 24
case median time ratios were below one, from 0.90261 to 0.99503; their median
was 0.97435. The maximum traced peak ratio was 1.00176. Thus this run passes
the unchanged development gates. Absolute baseline medians range from about
0.60 ms to 66.71 ms; the smallest timings remain sensitive to shared-host noise.
Do not present this observation as statistical significance or a default-auto
compiler speedup.

The [complete record](evidence/compiler-sabre-trial-dependencies.json) retains
every sample, source/driver text and digest, exact implementation and baseline
commits, host, limits and completion state. No failed timing case was dropped
or rerun. The 311 focused routing/conformance tests pass after restoring the
contracts directory omitted from the first sparse checkout. The earlier run
had 310 passes and one missing-file error; that setup attempt is not a pass.
Typing, semantics, protected API and full CI remain separate gates.

Additional verification passed six complete-operator QFT/QAOA/Ising application
cases plus benchmark import inventory and API candidate checks (22 tests).
Focused Black, Ruff, mypy, English, local links and changed-artifact revision
checks pass. The public API migration baseline passes against this source tree.
Its first setup attempts selected an installed wheel and then lacked proposal
files in the sparse checkout; neither is passing evidence. Expanding the owned
checkout and selecting its source resolved those environment errors without
changing a contract or snapshot. No capability support boundary changes.
