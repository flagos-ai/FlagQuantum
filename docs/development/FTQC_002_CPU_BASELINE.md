# FTS-02B: Initial CPU baseline observations

## Decision and scope

On 2026-10-24 the user explicitly selected baseline collection first and left the proposed 1.5x throughput and 2x capacity objectives disabled. This delivery records existing workflows. It adds no performance gate, public API, package, approximation strategy, hardware execution or capability promotion. Existing Dynamic wrapper budgets are unchanged and their original verdict is retained.

The measurement revision is main `89221bdb88aca82a017fcce27b49b9eb9fca33ac`, the merge of #805, including separately delivered AI calibration and Dynamic/Ising changes. Independent open #806 (Stim experiment bridge) and #807 (Ising actions) are not part of this source revision and are not duplicated here.

The reference scope is local CPU development evidence. It is not encoded non-Clifford FTQC, distributed capacity, domestic-device certification, threshold estimation, real-time feedback or experimental cross-validation.

## Measurement protocol

Reuse the existing construction producer, maintained Dynamic CLI and circuit-sampling/PyMatching user example. No second registry or measurement engine is introduced. A single controller invokes them sequentially from the clean exact source revision after the local regression/build processes finish. Command lines, versions, thread limits, timestamps, return codes, raw producer payloads and failures are retained in the accompanying evidence. Commands use one CPU thread; no GPU, remote submission, credentials or paid resources are involved.

W3 runs d=3/5/7, rounds=3, three host warmup processes followed by ten measured fresh processes per point. These warmups do not warm a reused in-process model; each producer reconstructs its own state. Tracemalloc construction time excludes earlier layout/lowering/imports. Whole-process RSS is a high-water mark, not exclusive DEM memory. Full subprocess wall time includes imports. Report medians and observed range without interpreting them as confidence bounds.

W2 runs repetition d=3 and Surface d=3/5/7 in X/Z readout, with 4096 fixed shots, batch size 32 and the delivered seed 53 stream. The producer owns the noise, mapping and shot-level failure definition; preserve every count and its 95% Wilson interval. These intervals concern logical counts, not throughput. Repeating an identical seeded stream is not an independent statistical replicate. No threshold fit or rare-event accuracy claim is made.

W5 retains the maintained runner's own three warmups, ten execution observations and ten planning observations, at 128 batched/static shots and eight trajectory shots. This intentionally small trajectory case does not establish large-scale or fault-tolerant control. Its raw timing boundaries and pre-existing budget verdict are preserved.

## Remaining work and acceptance boundaries

W1 closed-form H-T-H and analytic feedback invariants pass as public user examples; separate phase timings remain uncollected. W2 statistics are not phase-separated W6 compile/sample/decode/memory measurements. W3 d>=9 and larger/longer workloads remain unmeasured. W4 needs mixed-algorithm selection and its contract; W7 needs a named domestic device, stack and resource owner; W8 needs licensed raw experimental records and a data owner. These are open requirements, not successful empty cases. FTS-02 remains partial pending review of the selected baseline and its remaining measurement scope; the full M0-M5 mission is not closed.

Keep the existing numerical checks and protected QEC export/class/pickle/registration contracts. No original fixture is changed and #731 is untouched. Performance targets remain disabled until a later explicit decision.

## Observations

[Raw producer payloads and provenance](evidence/ftqc_cpu_baseline_20261024.json) retain all 47 subprocess cases, including nine designated host warmups. Values below are descriptive observations, not performance gates or relative improvements.

| W3 distance | Measured processes | Construction median (s) | Observed range (s) | Traced Python peak median (bytes) | Whole-process RSS range (bytes) |
| --- | --- | --- | --- | --- | --- |
| 3 | 10 | 0.174457 | 0.166408–0.211086 | 475848 | 200261632–217366528 |
| 5 | 10 | 1.069514 | 1.012180–1.204283 | 1387985 | 201555968–220446720 |
| 7 | 10 | 3.717929 | 3.333216–4.359893 | 2596205 | 219152384–225230848 |

| W2 code | Distance | Basis | Completed shots | Logical failures | 95% Wilson interval |
| --- | --- | --- | --- | --- | --- |
| RepetitionCode | 3 | z | 4096 | 25 | [0.004138, 0.008995] |
| RotatedSurfaceCode | 3 | x | 4096 | 24 | [0.003941, 0.008704] |
| RotatedSurfaceCode | 3 | z | 4096 | 17 | [0.002593, 0.006637] |
| RotatedSurfaceCode | 5 | x | 4096 | 3 | [0.000249, 0.002151] |
| RotatedSurfaceCode | 5 | z | 4096 | 2 | [0.000134, 0.001779] |
| RotatedSurfaceCode | 7 | x | 4096 | 0 | [0.000000, 0.000937] |
| RotatedSurfaceCode | 7 | z | 4096 | 0 | [0.000000, 0.000937] |

W5 planning median is 0.000113646 s, p95 0.000141681 s across ten retained observations. The unchanged producer reports `budget_passed=true`. This is its existing wrapper-budget verdict, not the newly proposed throughput/capacity targets. Inspect `dynamic_cases` and `static_sentinel` for all raw execution observations and correctness checks.

Environment: Apple M5; macOS-27.0-arm64-arm-64bit; Python 3.12.14, Torch 2.13.0, Stim 1.16.0, PyMatching 2.4.0; one Torch/OMP/MKL/OpenBLAS thread. Measurement timestamps and source tree are in the artifact. This source measurement environment differs from the clean-install default Torch 2.14.1 acceptance profile. Neither is relabeled as the other.

## Runnable public-user example

Core CPU installation is sufficient for this analytic interference and feedback example; no optional SDK, GPU, credentials or network job is needed. Save the following as `/tmp/ftqc_reference.py`, then run `python /tmp/ftqc_reference.py` with the installed reviewed wheel, or `PYTHONPATH=. python /tmp/ftqc_reference.py` from this checkout.

```python
import math

import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

program = fq.Circuit(1).h(0).t(0).h(0)
options = fq.ExecutionOptions(
    mode="statevector", device="cpu", precision="complex128"
)
probabilities = fq.run(
    program, outputs=fq.probabilities([0]), options=options
).probabilities
assert probabilities.shape == (1, 2)
expected = torch.tensor(
    [(2 + math.sqrt(2)) / 4, (2 - math.sqrt(2)) / 4], dtype=torch.float64
)
assert float((probabilities[0] - expected).abs().max()) <= 1e-10
print("HTH probabilities:", *[f"{float(p):.8f}" for p in probabilities[0]])

circuit = DynamicCircuit(2).h(0).measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)
result = fq.run(circuit, options=fq.ExecutionOptions(shots=128, seed=53))
assert result.samples is not None and result.samples.shape == (128, 2)
print("Feedback parity:", bool((result.samples[:, 0] == result.samples[:, 1]).all()))
```

Expected stdout:

```text
HTH probabilities: 0.85355339 0.14644661
Feedback parity: True
```

The probability check reuses the proposal's existing reference example; it is not a new general numerical contract. Bit parity alone does not certify a complete Born distribution or a fault-tolerant logical operation.

## Reproduce producer observations

Use a clean exact checkout, core Torch and the `stim`/`pymatching` extras. These commands are existing workflows. Bound each invocation (120 s for the sampling/dynamic command, 300 s for one construction point) and retain failures. This short reproduction is one observation; the artifact's W3 methodology uses three host warmups and ten fresh measured processes.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
python benchmarks/qec_clifford_dem_construction.py \
  --distance 3 --rounds 3 --output /tmp/ftqc-dem.json

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
python examples/qec/stim_pymatching_baseline.py \
  --code surface --distance 3 --basis x --shots 4096 --batch-size 32

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
python -m flagquantum.benchmarking run dynamic_unified_run \
  --batched-shots 128 --trajectory-shots 8 --static-shots 128 --seed 53 \
  --warmup 3 --iterations 10 --planning-iterations 10 \
  --json-output /tmp/ftqc-dynamic.json
```

Expected invariants: construction has `distance=3`, `rounds=3`, and scalability/release flags false; the Surface report has `sampling="circuit"`, `readout_basis="x"`, `shots=requested_shots=4096`, `batch_size=32`, one observable and `stopping_rule="fixed_shots"`; Dynamic raw sample counts match its requested repetition count and its native/automatic/explicit correctness checks pass. Timings, RSS, logical failures and Wilson intervals are observations, not universally pinned outputs.

## Post-#805 main acceptance

The exact measured main `89221bdb88aca82a017fcce27b49b9eb9fca33ac` passes [complete CI, 54/54 jobs](https://github.com/flagos-ai/FlagQuantum/actions/runs/40082974065), [source/single-/required two-GPU checks](https://github.com/flagos-ai/FlagQuantum/actions/runs/40082984925) and [smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/40082402550). Three existing optional manual GPU-pool jobs were unselected. The local source QEC, retirement-contract and both Simulation fault-signature suites pass with **2,120 passed, one original inapplicable-case skip, 385.86 seconds**. None of these results is inferred from an older revision or a new skip.

Fresh isolated sdist and wheel-from-sdist contain all **821 package Python members** byte-identical to this main. A clean Python 3.12.14 core install resolves Torch 2.14.1, passes dependency/native/Bell checks outside the checkout, and runs both master-plan public examples and the H-T-H/feedback code above with their expected stdout. This is one fresh core install; exact-head CI supplies the current broader Python/SDK coverage. No historical local twelve-environment rerun is claimed.

| Artifact | SHA-256 |
| --- | --- |
| `flagquantum-0.4.0.tar.gz` | `7a727b71ec737466a7567dc20cece288ddf1ae0dca77d81f7b1d45f8842fb76e` |
| `flagquantum-0.4.0-cp310-abi3-macosx_11_0_arm64.whl` | `8df2eb3e41abeba9ad2e8c5852e0331d0d9293bdbc8933139035989248dced86` |

This acceptance covers the merged source, including separately delivered AI calibration, Dynamic routing and Ising source changes. The documentation-only #805 merge is not credited with their implementation. This follow-up changes documentation and retained measurement evidence only; public exports, class/pickle identities, numerical code, registries and all original frozen fixtures are unchanged. No version is published.
