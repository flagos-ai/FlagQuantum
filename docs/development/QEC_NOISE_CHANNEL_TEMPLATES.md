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
API checks passed. The required default tier completed with 11,420 passed,
85 skipped, four native-fusion expectation failures and four Gloo setup errors in
187.09 seconds, with the configured failure cutoff. A source archive of the exact
unmodified main, with the imported package path asserted, reproduced those four
failures and six errors in the complete focused Gloo module (237 passed).
The local default suite is not green; final delivery gates remain pending.

Private cold profiling motivated investigation of repeated preparation work.
Those instrumented observations are attribution only. There is no measured
uninstrumented performance result for this candidate yet, no speedup claim and no
new full-distance, GPU, threshold, statistical-significance or capacity evidence.
A bounded comparison must freeze the implementation and protocol before timing;
all results and any regressions must be retained.

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
