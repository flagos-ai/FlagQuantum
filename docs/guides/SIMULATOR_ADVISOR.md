# Evidence-based simulator advisor

FlagQuantum can rank explicit local simulator choices from checked-in comparison
evidence without changing execution routing. Circuit-bound queries require the
canonical `CircuitIR.content_hash` to match a versioned workload manifest as
well as the recorded environment. The first evidence bundle covers the exact
`hardware_efficient_statevector` circuits measured on the recorded
single-process ARM64 CPU environment:

```python
from flagquantum.benchmarking.simulator_compare import build_workload
from flagquantum.ecosystem.simulators import recommend

measured_circuit = build_workload(n_wires=22, layers=2)
decision = recommend(measured_circuit)

print(decision.status)
print(decision.recommended_engine)
print(decision.circuit_matches)
print(decision.circuit_ir_hash)
for candidate in decision.candidates:
    print(candidate.engine, candidate.median_seconds, candidate.eligible)
```

The result contains the measured median for every candidate, its ratio to
FlagQuantum, measured and installed versions, availability and stability
filtering, the workload fingerprint, the source artifact and checksum, and the
reasons for any exclusion. It also records the workload-manifest path and
checksum for an exact circuit match.

The `build_workload` import above reproduces the benchmark circuit; it is not
required for ordinary FlagQuantum programs. For an arbitrary user circuit,
call `recommend(circuit)` directly. Without an explicit calibration budget, an
unmeasured canonical IR hash returns `insufficient_evidence` with reason
`circuit_not_measured`—even when its width and gate count happen to match a
measured circuit.

## Calibrate an arbitrary circuit

An application can explicitly permit a short live comparison when checked-in
evidence does not contain the circuit:

```python
decision = recommend(
    circuit,
    calibration_budget_seconds=2.0,
    calibration_warmup=1,
    calibration_repeats=5,
)

print(decision.evidence_level)  # "exact" or "live_calibration"
print(decision.confidence)
print(decision.calibration_elapsed_seconds)
print(decision.calibration_cache_hit)
```

Live calibration executes the exact supplied circuit as a statevector on the
native runtime and installed external bridges. Native output is the correctness
reference. A candidate needs a matching statevector, enough retained samples,
and relative median absolute deviation no greater than 25%. The usual native
tie margin is then applied to the measured medians.

The budget is a soft wall-clock target checked between backend calls. A single
framework import, compilation, or backend call is not interrupted and can
overrun it. Later candidates are marked `calibration_budget_exhausted` once the
budget has elapsed. Restrict `available_engines` when only particular installed
frameworks should be probed. Results are cached in the current process by
circuit hash, platform, Python and Torch environment, engine versions, and
calibration settings.

This explicit live path avoids pretending that the initial single workload
family can train a reliable performance predictor. A future similarity model
requires a broader benchmark corpus and independently validated error bounds;
until then, unknown circuits are measured or rejected rather than guessed.

### Live calibration validation probe

The implementation was exercised on an unmanifested 10-qubit circuit containing
per-wire H, RZ and RX gates followed by a nearest-neighbor CX chain. On the same
ARM64 macOS environment as the checked-in comparison, using Python 3.12.14,
Torch 2.13.0, FlagQuantum 0.2.0 and Qiskit Aer 0.17.2, two warmups and seven
retained calls produced:

| Engine | Median | Relative to FlagQuantum | Relative MAD | Eligible |
| --- | ---: | ---: | ---: | --- |
| FlagQuantum | 3.252 ms | 1.00x | 4.29% | yes |
| Qiskit Aer | 29.161 ms | 8.97x slower | 17.37% | yes |

The advisor selected FlagQuantum with confidence `0.90`; total calibration time
was 0.868 seconds. Cirq and PennyLane were intentionally left outside this
probe through `available_engines`. These are development validation numbers for
that exact circuit and environment, not a universal performance claim.

Profile-only queries remain available for inspecting the table:

```python
profile = recommend(n_wires=22)
assert profile.circuit_matches is None
```

Such a result is not bound to a user circuit and must not be used as evidence
for automatic routing.

The default evidence-only path never executes a circuit or imports an optional
simulator. Only an explicit positive `calibration_budget_seconds` enables live
execution. Neither path changes `fq.run`, automatically routes execution, or
silently falls back. After inspecting a recommendation, select an execution
bridge explicitly:

```python
from flagquantum.ecosystem.qiskit import run as run_qiskit

result = run_qiskit(circuit)
```

## Conservative decision policy

- Circuit IR hash, workload fields, and the recorded environment must match
  exactly for a circuit-bound recommendation. The advisor does not substitute
  width or gate count for circuit identity, interpolate between qubit counts,
  or extrapolate to another CPU.
- Incorrect, unstable, unavailable, or version-mismatched candidates are
  excluded.
- An unknown circuit is never inferred from width or gate count. It remains
  insufficient evidence unless the caller explicitly enables live calibration.
- If an external simulator is no more than 10% faster than native execution,
  the advisor recommends FlagQuantum to avoid extra dependency and conversion
  cost. The threshold is configurable through `native_tie_margin`.
- The bundled comparison excludes conversion and compilation from steady-state
  timing. The returned limitation text keeps that evidence boundary visible.
- Live calibration measures end-to-end bridge calls, including conversion and
  compilation performed by each call, after the requested warmup.
- The measurements are non-release comparison evidence, not a universal
  ranking, production guarantee, or scalability claim.

Use `environment` and `available_engines` to reproduce a decision in tests or
offline analysis. Use `evidence` to supply a newer report mapping or JSON path;
malformed reports, hidden-fallback evidence, failed correctness, and promoted
release claims fail closed. Custom evidence used with `recommend(circuit)` must
put a lowercase SHA-256 `ir_content_hash` on the matching report row.

## Reproduce the workload identity binding

The versioned manifest is
`benchmarks/manifests/simulator_workload_ir_v1.json`. It references the original
comparison report and its checksum without changing any timing value. The
focused test regenerates all five deterministic circuits, checks their IR
hashes against the manifest, checks each manifest workload and fingerprint
against the original report, and verifies both file checksums:

```bash
pytest -q tests/test_simulator_advisor.py
```

This binding was added after the original benchmark run because that timing
artifact did not record circuit IR hashes. It proves that current deterministic
builder output corresponds to each existing row; it does not claim that the
original artifact captured a source commit or IR hash that it did not capture.

## Bundled measured decisions

The initial ARM64 artifact produces the following default decisions. Times are
steady-state medians from the checked-in report; they do not include conversion
or compilation.

| Qubits | FlagQuantum | Fastest external | External time | Decision |
| ---: | ---: | --- | ---: | --- |
| 10 | 0.577 ms | PennyLane Lightning | 0.554 ms | FlagQuantum; external gain is only 1.04x |
| 14 | 1.307 ms | PennyLane Lightning | 1.488 ms | FlagQuantum; 1.14x faster |
| 18 | 13.125 ms | PennyLane Lightning | 16.484 ms | FlagQuantum; 1.26x faster |
| 22 | 300.297 ms | PennyLane Lightning | 341.139 ms | FlagQuantum; 1.14x faster |
| 24 | 1,164.636 ms | PennyLane Lightning | 1,610.489 ms | FlagQuantum; 1.38x faster |

Cirq is excluded from the 10-, 14-, and 18-qubit decisions because those rows
exceeded the declared relative-median-deviation stability threshold. Qiskit Aer
remains eligible but was not the fastest stable engine in these measured rows.
