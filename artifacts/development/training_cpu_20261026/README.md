# Matched CPU training evidence

This retained development campaign supports the CPU training portion of
[#900](https://github.com/flagos-ai/FlagQuantum/issues/900). It establishes
matched objectives, independent initial value/gradient agreement, fixed-budget
training, time-to-target and optimizer recovery. It is not release evidence.

See [the generated report](REPORT.md) and
[the protocol](../../../docs/development/TRAINING_COMPARISON_PROTOCOL.md).
The immutable producing commit is
`a6c634e9317d7f08cfa4053c6fbd712dabfdd99f`, with a clean working tree for
every case. `freeze.json` binds the initial parameter vectors, data hash,
source hashes, protocol and package versions. Each case retains its raw JSON
and process log. Published JSON copies redact infrastructure identity using
the repository sanitizer; `publication.json` records original and published
file hashes. Local originals remain unchanged. Measurements used macOS arm64 CPU, one configured thread,
complex128 amplitudes and float64 parameters.

All 33 cases completed: three paired seeds, five quantum backend/derivative
routes for each of two tasks, and a six-parameter classical classifier.
Maximum initial observable error was 4.45e-15; maximum independent
finite-difference gradient error was 2.81e-10. Final parameters, gradients and
Adam slots were reproduced exactly after checkpoint reload in every case.

All quantum engines reached the same final task scores for each seed.
Classifier test accuracy was 32/32 on this small, fixed held-out split;
Heisenberg relative energy error ranged from 1.2318e-4 to 1.9207e-4.
The quadratic classical control scored 29/32 and was much cheaper. These
observations do not establish generalization or quantum advantage.

FlagQuantum's median 60-update training times were 2.4000 seconds for the
classifier and 0.2927 seconds for Heisenberg, including validation after every
update. Both were faster than the measured PennyLane backprop and
TensorCircuit-NG Torch paths. Lightning adjoint was faster on Heisenberg.
TensorCircuit-NG JAX was faster after compilation on both tasks, with first
value/gradient costs of about 0.11 and 0.74 seconds, respectively. Backend
setup, first-call, warmup, training, final assessment and diagnostic measured
phase totals are retained separately in JSON. Optional backend imports occur
inside setup while common imports precede it; diagnostic totals are not fair
cold-start rankings. This serial-example corpus does not compare each
framework's best batching configuration.

## Reproduction and verification

From the producing commit and resolved environment:

```bash
python -m benchmarks.training_comparison --output /tmp/training-cpu-new
python -m benchmarks.training_comparison_report /tmp/training-cpu-new
```

The committed report can also be regenerated from the retained JSON. Binary
checkpoints remain outside the source repository, as required by the
development-artifact policy; each result records its checkpoint hash.
Rerunning the producer writes fresh checkpoints and repeats recovery checks.

Verification completed during delivery:

- 15 scientific parity/recovery tests passed with all five optional quantum
  engines present; two classifier/VQE example integration tests passed.
- Black, Ruff, mypy, architecture, dependency and team-scope checks passed.
- `pr-default` did not pass: its full run exceeded a 300-second budget and
  was terminated with its process group. Before termination, the installed
  entry-point discovery test failed because this environment imports the
  source checkout without installing its distribution metadata. An isolated
  rerun reproduced `declared [('backend', 'qiskit_aer')] but discovered []`
  (47 preceding tests passed). This is an environment limitation, not a
  claim that the default suite is green.

The earlier adjoint preflight rejected float32 output caused by Torch's
default dtype. The final producer fixes that default at float64 and retains
the original parity tolerances. Failed exploratory outputs remain separate
from this frozen 33-case campaign.

QNG algorithm/fusion separation, independent parameter scaling, larger tasks,
repeated timing campaigns and accelerator measurements remain open in #900.
