# Native CPU adjoint and public training time to quality

This is a single-host CPU comparison for #936 and #918. It uses the frozen
classifier and four-site Heisenberg tasks, seeds 11, 29 and 47, FP64 parameters,
complex128 amplitudes, and the original Adam settings and quality thresholds.
Each route/task/seed runs in a fresh process with one Torch thread, alternating
route order by seed. All 12 runs completed within the 240-second case limit.
The three measurements per route use different seeds and describe this
campaign's dispersion rather than repeatability at a fixed seed.

| Task | Route | First target step by seed | Cold time to target, median (range) | Cold total for 60 updates, median |
| --- | --- | --- | --- | --- |
| Classifier, validation accuracy ≥ 0.80 | Public fq.train | 24, 24, 21 | 0.913 s (0.897–1.224 s) | 0.958 s |
| Classifier, validation accuracy ≥ 0.80 | Native CPU adjoint | 24, 24, 21 | 2.470 s (2.386–2.890 s) | 5.398 s |
| Heisenberg, relative energy error ≤ 0.01 | Public fq.train | 21, 18, 10 | 1.012 s (0.992–1.053 s) | 1.225 s |
| Heisenberg, relative energy error ≤ 0.01 | Native CPU adjoint | 21, 18, 10 | 1.473 s (1.396–1.542 s) | 2.622 s |

The public leg uses an ordinary Torch-default Module with a callback for
per-update quality measurements. Classifier training uses the public declared
batch of 64 samples. The native leg runs the existing local CPU
statevector reverse executor with ordinary Adam. The classifier's batched
native probe refused a 32-element RY feature angle at the forward executor,
which expected a scalar at batch size one. The measured native classifier
therefore runs 64 serial sweeps per update. This protocol measures route-level
latency, with different execution granularities for classifier training.
For Heisenberg the native executor accepts weighted Z/ZZ, so three measurement
bases implement the exact XX/YY/ZZ Hamiltonian. The public route consumes that
Hamiltonian directly. Neither comparison isolates kernel speed.

Both routes use the same batched no-grad Module validation evaluator.
The timer starts at worker entry before framework imports and includes setup,
initial validation, updates, per-update validation and any checkpoint write
before the threshold crossing. The update-30 save is included in the fixed
60-update total. The held-out assessment, checkpoint replay and independent
dense parity checks run after the timed training. The process peak RSS covers
all phases. The median includes one observation per seed. The receipt keeps
all update and validation times, setup, import, checkpoint and replay times,
plus process RSS.

The native backward evidence reports 3,840 completed sweeps and 15,360 fused
parameter-adjoint operations per classifier run, and 180 sweeps and 2,520 fused
parameter-adjoint operations per Heisenberg run. This is single-process
execution. Each route's fresh optimizer replay checks final parameters,
gradients and Adam slots at absolute tolerance 1e-10. Across routes, the largest
per-update loss and validation-score differences were 1.15e-14 and 1.65e-15.
The maximum final parameter difference was 1.09e-8. Independent dense checks
were within 5.33e-15 in value and 2.81e-10 in gradient. Public consumed IR hashes
are retained per step. The native comparator uses a different circuit
granularity and observable basis, so the receipt does not assert cross-route
IR hash equality.

## Reproduce

Run from the repository root with the built native CPU extension in the same
checkout. The campaign ran on Apple M5 with Python 3.12.14 and Torch 2.14.1.
The receipt pins source commit
`0fe2e2cea541273bc371c98482a087b4cdef0313` from merged PR #1022, the protocol
and benchmark source digests, and the native extension digest. The checkout
was marked dirty because the evidence directory was present. The tracked
runtime and benchmark files matched the recorded commit.

```bash
D=artifacts/development/native_adjoint_training_cpu_20261103
python "$D/run.py" --output /tmp/fq-native-adjoint-training.json
python "$D/verify.py" /tmp/fq-native-adjoint-training.json
python "$D/verify.py"
```

The campaign retains failed and timed-out processes in the output. Raw
checkpoint files are temporary. The verifier checks this campaign's expected
successful routes, quality crossings and native execution evidence.

## Remaining acceptance

- **#936** now has a public Torch versus actual native-adjoint time-to-quality
  comparison for the two original tasks. The earlier held-out shape matrix
  reports cold Inductor losses and a separately measured external whole-JAX
  control. The compiled route still needs a default-CPU end-to-end benefit,
  matched time-to-quality against a strong whole-training comparator on the
  held-out matrix, and measured invalidation/recompile costs. Adam updates
  still execute eagerly. The callback benchmark here runs the ordinary eager
  path and measures no compiled route improvement.
- **#918** now has fresh-process cold quality and native execution data for the
  original classifier and Heisenberg tasks. It still needs a matched public
  comparison with the strongest applicable supported routes under comparable
  batching, repeated timing trials, and demonstrated default-path
  accessibility. The related declared-batch, explainable selection and
  finite-angle validation issues #929, #930 and #925 remain open. The A800
  checkpoint correctness run from #1022 does not measure GPU time to quality.
