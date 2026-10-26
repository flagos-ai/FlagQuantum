# Local noise-channel construction templates

This private CPU preparation candidate follows the prepared parity work in
[QEC_PREPARED_CPU_PARITY.md](QEC_PREPARED_CPU_PARITY.md). It changes only
phenomenological channel construction in `_noisy_program`; it does not add a
public option, numerical kernel, SDK adapter or sampling law.

Each noise location still produces its own instruction at its original position,
with the original wires, parameters and complete factory-produced Kraus channel.
Separate locations remain independent random events. Equal channel factories and
rates can reuse a calculation within this one program construction, keyed also by
runtime complex precision and PyTorch's default scalar precision and device.
The original factory performs probability conversion, rounding and validation.
No nominal probability or channel name replaces the positioned operators.

The template dictionary is local to one call and is discarded on return or
exception. Every instruction receives fresh tensor clones: `Instruction` retains
matrix references, and a frozen record does not make a tensor immutable. Direct
sharing would make mutations at one location affect another. There is no global,
SDK, cross-program or cross-handle cache. Gate-bound `NoiseModel` placement and
Simulation's operator classification are unchanged.

## Validation and limits

The original implementation passed the semantic checks but failed the repeated
factory-call regression: eight locations invoked their factories eight times,
where four distinct factory/rate combinations suffice. The candidate passed
22 focused cost, bitwise-operator, tensor-ownership, ordering and precision cases.
It also passed 177 related tests with real optional SDKs, including existing noise,
memory sampling, stream and prepared-parity coverage. Another 116 bridge, strict
DEM precision, Surface and fixed-shot tests passed with real SDKs. All 22 new
tests also passed in a core-only environment where Stim/PyMatching were absent.
Strict typing of the
modified production module, Black, Ruff, architecture, dependency and read-only
API checks passed. The CI-style whole-package strict mypy command also passed
834 source files with incremental caching disabled and the documented SDK/stub
environment; this does not certify other environments. The required default tier completed with 11,420 passed,
85 skipped, four native-fusion expectation failures and four Gloo setup errors in
187.09 seconds, with the configured failure cutoff. A source archive of the exact
unmodified main, with the imported package path asserted, reproduced those four
failures and six errors in the complete focused Gloo module (237 passed).
The local default suite is not green; final delivery gates remain pending.

## Bounded local comparison

[Protocol, observer and complete raw observations](evidence/qec_noise_channel_templates_20261026.json)
pin exact main `ebc2f4985eaaa5e40b81f426dad4e872fa3f559b` and frozen
implementation `3ec3d3e949d9717f2ad35601ece2ea660011453e`. Source and tests
remain unchanged in the later evidence commit. Earlier private cold profiles
motivated investigation only; their instrumented times are not speed baselines.

This independent local macOS arm64 CPU comparison used Python 3.12.14,
Torch 2.13, Stim 1.16 and PyMatching 2.4, one numerical thread, d9 X memory,
three rounds, 65,537 shots, seed 53 and noise .005/.007/.009. Each batch used
six fresh processes in baseline/candidate, candidate/baseline, baseline/candidate
order. Each process retained one warmup and three measured rows. Limits were
30 seconds per process and 360 seconds total, with no concurrent local tests.
All 12 processes completed in 82.50771399997757 seconds, with no retries,
additional samples or seed changes. Nine measured rows per implementation and
batch are repeated same-seed observations, not independent statistical trials.

| Batch | Baseline pipeline median (s) | Candidate pipeline median (s) | Baseline loop median (s) | Candidate loop median (s) |
| --- | --- | --- | --- | --- |
| 128 | 1.274472167 | 1.193746292 | .241406199 | .243828384 |
| 4,096 | 1.213612250 | 1.159226583 | .189791663 | .189508785 |

Pipeline ratios are 1.0676 and 1.0469: descriptive differences of about 6.3%
and 4.5% less wall time for this fixed workload. The small-batch loop is about
1% slower in this observation; the large-batch loop is nearly unchanged. This
preparation-only change does not claim loop acceleration, statistical significance
or general throughput benefit. No earlier study ratios are pooled or multiplied.
Pipeline includes memory construction, export, strict actual-law DEM, decoder
setup, sampler compilation, sampling/decoding/count reduction and close. Imports,
metadata and independent full-shot diagnostics are excluded equally. All warmups
are retained and excluded from medians.

All six processes per batch have identical source, strict DEM, detector,
observable and prediction SHA256 and event/logical counts. Diagnostics include
the one-shot tail: batch128 records 512,157 detector events and two logical
mismatches; batch4096 records 513,809 and zero. Cross-batch bit identity is not a
seed contract, and these counts are not suppression or threshold evidence.
Whole-process RSS includes imports, warmup and diagnostics: roughly 267–269 MB
at batch128 and 284–286 MB at batch4096, with overlapping ranges. It is not a
phase-specific peak or capacity certificate. No new full-distance matrix, GPU,
threshold, release performance or state-capacity result is claimed. Public
host/source paths are anonymous; numerical rows, warmups and original raw-file
SHA256s are retained.

The existing nominal-decimal versus finite-precision Kraus limitation remains.
Strict actual-law interoperability retains its existing tolerance and original
source sampler; a decoder graph is not a replacement sampling distribution.

## Existing user workflow

Install `python -m pip install -e '.[stim]'`. No GPU is required:

```python
from flagquantum.qec import (
    PhenomenologicalNoise, RepetitionCode, build_memory_circuit,
    compile_memory_sampler,
)

memory = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
with compile_memory_sampler(memory, noise=PhenomenologicalNoise(), seed=53) as sampler:
    for shots in (8, 8, 1):
        batch = sampler.sample(shots=shots)
        assert batch.detectors.shape == (shots, 8)
        assert batch.observables.shape == (shots, 1)
        assert not batch.detectors.any() and not batch.observables.any()
print("17 shots, no noiseless flips")
```

Expected output: `17 shots, no noiseless flips`. This example demonstrates the
unchanged public workflow, not a performance experiment. It was executed unchanged
on local Python 3.12.14/Torch 2.13/Stim 1.16 and printed the expected line.
