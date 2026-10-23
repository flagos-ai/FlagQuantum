# QEC execution composition

This directory composes existing QEC code/noise records with experiment runners.
It currently owns only `noise_sweep.py`; repetition execution, sampling and
forced-signature migration are later batches. It must not own code definitions,
noise records, numerical kernels or another registry. It may consume QEC-owned
records and the existing repetition experiment. Its package initializer imports
no experiment machinery.

`noise_sweep.py` contains the original finite-shot loop without numerical or RNG
changes. Its public function remains the same object at `flagquantum.qec`,
`flagquantum.qec.noise` and this implementation module. The historical module
name stays `flagquantum.qec.noise` for function pickles. The noise module resolves
this alias lazily so its records do not import an executor. The existing QEC
root facade remains eager; this batch does not claim cold root imports avoid
compiler/runtime.

Core owns these compatibility aliases. No removal release is approved; changing
the public import or pickle path requires a separate approved proposal.

A short local workflow:

```python
from flagquantum.qec import run_repetition_memory_noise_sweep

points = run_repetition_memory_noise_sweep((0.0,), rounds=3, shots=8, seed=53)
print(points[0].logical_error_rate)
# 0.0
```

For a sweep change, edit `noise_sweep.py`, preserve validation, profile mapping,
seed offsets and result construction, then run:

```bash
python -m pytest tests/qec/test_restructure_boundaries.py tests/qec/test_repetition_memory.py tests/qec/test_phenomenological_noise.py -q
python examples/qec/restructure_compatibility.py
```
