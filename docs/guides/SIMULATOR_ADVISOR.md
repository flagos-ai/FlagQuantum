# Evidence-based simulator advisor

FlagQuantum can rank explicit local simulator choices from checked-in comparison
evidence without changing execution routing. The first evidence bundle covers
the exact `hardware_efficient_statevector` workload measured on the recorded
single-process ARM64 CPU environment:

```python
from flagquantum.ecosystem.simulators import recommend

decision = recommend(n_wires=22)

print(decision.status)
print(decision.recommended_engine)
for candidate in decision.candidates:
    print(candidate.engine, candidate.median_seconds, candidate.eligible)
```

The result contains the measured median for every candidate, its ratio to
FlagQuantum, measured and installed versions, availability and stability
filtering, the workload fingerprint, the source artifact and checksum, and the
reasons for any exclusion.

The advisor never executes a circuit, imports an optional simulator, changes
`fq.run`, or silently falls back. After inspecting a recommendation, select an
execution bridge explicitly:

```python
from flagquantum.ecosystem.qiskit import run as run_qiskit

result = run_qiskit(circuit)
```

## Conservative decision policy

- Workload fields and the recorded environment must match exactly. The advisor
  does not interpolate between qubit counts or extrapolate to another CPU.
- Incorrect, unstable, unavailable, or version-mismatched candidates are
  excluded.
- If an external simulator is no more than 10% faster than native execution,
  the advisor recommends FlagQuantum to avoid extra dependency and conversion
  cost. The threshold is configurable through `native_tie_margin`.
- The bundled comparison excludes conversion and compilation from steady-state
  timing. The returned limitation text keeps that evidence boundary visible.
- The measurements are non-release comparison evidence, not a universal
  ranking, production guarantee, or scalability claim.

Use `environment` and `available_engines` to reproduce a decision in tests or
offline analysis. Use `evidence` to supply a newer report mapping or JSON path;
malformed reports, hidden-fallback evidence, failed correctness, and promoted
release claims fail closed.

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
