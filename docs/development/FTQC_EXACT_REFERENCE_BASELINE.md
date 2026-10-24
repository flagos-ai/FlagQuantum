# W1-A Exact Reference Phase Baseline

## Scope and delivery

This FTS-02D baseline observes the existing one-qubit H-T-H CPU statevector path. It fills W1-A construction/planning/execution timing only. It introduces no mixed simulator, QEC API, approximation, numerical tolerance or performance gate. The user's baseline-only decision remains in force; proposed 1.5x/2x targets remain disabled. FTS-02 and the mission-level mixed-engine requirement remain partial.

The observer uses the maintained lazy benchmark registry, runtime metadata and atomic report writer. Existing dynamic feedback timing and compiled QEC sampling observers remain separate workload classes. Independent #819 installed optional profiles, #818 sampled Ising proposal and #812 hyperedge/Ising comparisons are not duplicated.

## Post-817 mainline acceptance

Main `d744e0efc74fa789817230d757e4ec4deed50c26` passed [complete CI, 54 jobs](https://github.com/flagos-ai/FlagQuantum/actions/runs/40096939172), [source/single/required two-GPU validation](https://github.com/flagos-ai/FlagQuantum/actions/runs/40096849771) and [smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/40096687895). Three existing optional GPU pools were unselected. Source QEC, frozen retirement, fault signatures, Stim, Ising, benchmark CLI and Quafu normalization selection: **2,316 passed, one original inapplicable-case skip**, 242.45 seconds.

Fresh isolated sdist and wheel-from-sdist preserve all **825 package Python members** exactly. A clean Python 3.12.14 / Torch 2.14.1 core install passes dependency consistency, native/Bell sampling and both master-plan user examples outside the checkout. The installed 17-shot sampling example and benchmark CLI additionally pass using the existing Stim 1.16.0 / PyMatching 2.4.0 SDK overlay; this overlay is explicitly not a fresh optional-profile install. No historic local 12-profile run is relabeled.

Artifact SHA256:

- sdist: `b377ab48a9ff77f2ad233f7240b10febf0503e88232b84fb192a7820988e9597`
- wheel: `08bd8f35a41f404f5a13a04fe92f16e18784d67dd38bed2774b9b974f8b2572b`

## Measurement protocol and observations

[All 39 raw observations](evidence/ftqc_exact_reference_20261024.json) bind clean obtainable producer `b4f4099844e921b062f40d4699ca2374fd58f30b`, a direct descendant of accepted main. Subsequent commits add documentation/data and the standard benchmark-contract test-selection marker; the measured observer code is unchanged. Three fresh processes each retain three labelled warmups and ten measured repetitions. CPU arm64/macOS, Python 3.12.14, Torch 2.14.1, one Torch thread, OMP/MKL/OpenBLAS thread environment one; no optional SDKs. Original per-process reports and stdout are retained locally; promoted reports remove only hostname, with original file digests retained. No numerical payload is rounded or replaced in the JSON.

Construction includes circuit/options construction. Planning includes probability-output declaration and `fq.plan`. Explicit execution is `fq.run(plan)`. Its total includes construction, planning and execution. Automatic execution separately includes output declaration and `fq.run(circuit, ...)` planning. Initial runner imports and report I/O are excluded; lazy imports inside called APIs remain included. Probability extraction, validation and analytic-error reduction occur after the timed calls. Explicit and automatic paths share runtime caches and execute in that order: these are observations, not an unbiased paired speed comparison. Warmups are retained but excluded from medians; process startup is recorded separately. CPU work is synchronous. No process RSS/capacity measurement is made here.

Medians in seconds, not performance commitments:

| Process | Construction | Planning | Explicit execution | Automatic execution | Explicit full pipeline |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.000021479 | 0.000285646 | 0.000597813 | 0.000830792 | 0.000916791 |
| 2 | 0.000021542 | 0.000263853 | 0.000598104 | 0.000819605 | 0.000880563 |
| 3 | 0.000021876 | 0.000272084 | 0.000637333 | 0.000849791 | 0.000932480 |

The analytic reference is `(2 + sqrt(2))/4, (2 - sqrt(2))/4`. Each repetition reports both full probability vectors, maximum absolute errors and normalization errors. These errors are observations, not newly approved tolerances. Malformed/nonfinite/wrong-precision results fail; execution errors propagate and no success artifact is written. The exact public example's interference check does not certify large mixed Clifford/non-Clifford simulation or an encoded fault-tolerant protocol.

## Runnable public user example

Core installation only, CPU; no Stim, PyMatching, hardware account or credentials. Save as `/tmp/ftqc_exact_reference.py` and run `python /tmp/ftqc_exact_reference.py` against an installed distribution. Source users use `PYTHONPATH=. python /tmp/ftqc_exact_reference.py` from the checkout.

```python
import flagquantum as fq

circuit = fq.Circuit(1).h(0).t(0).h(0)
options = fq.ExecutionOptions(mode="statevector", device="cpu", precision="complex128")
plan = fq.plan(circuit, outputs=fq.probabilities([0]), options=options)
probabilities = fq.run(plan).probabilities
print("HTH probabilities:", *[f"{float(p):.8f}" for p in probabilities[0]])
```

Expected stdout:

```text
HTH probabilities: 0.85355339 0.14644661
```

Run the observer from the source checkout:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=. python -m flagquantum.benchmarking run ftqc_exact_reference \
  --warmup 3 --iterations 10 --json-output /tmp/ftqc-exact-reference.json
```

Expected invariants: 13 observations, first three labelled warmup, two probabilities per execution path, CPU complex128/float64 precision, non-release/single-process classification, performance/scalability claims disabled. Timing/error values are observed, not pinned. The total explicit pipeline equals the sum of its three contiguous phases. WORLD_SIZE other than one and invalid repetition envelopes fail before measurement.

## Remaining acceptance gaps

This work advances measurement for R01/R03/R04/R10; none is declared fully satisfied by this one-qubit reference. W1-B fault/channel coverage, larger W3/W6 scale, mixed-engine selection, domestic device qualification and experimental data remain open. Matrix/resource/tolerance approvals, FTS-05 public batch semantics and FTQC package admission remain distinct approval gates. The original QEC exports, class/pickle identity, numerics, registry and frozen fixtures are unchanged.
