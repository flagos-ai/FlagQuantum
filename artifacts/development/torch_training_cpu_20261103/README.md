# CPU Torch training candidate: preserved cold failures and public-path improvement

This packet supports the user-approved #936 implementation boundary in #918.
**Neither issue is ready for closure.** The compiled candidate is not enabled by
default: on this reserved 60-update corpus its total training time is 25.0-49.8
times the eager callback route. Adam optimizer updates remain eager. There is no
whole-step fusion, automatic JAX selection, GPU performance or scalability claim.

The separate default improvement is available through ordinary `fq.train`:
steps without a callback reuse Module's existing tensor-only forward path;
the final result and every callback still receive complete execution metadata.
Consumed pre-update IR remains readable and checkpointable between updates.
This removes unused result work without changing numerics or public signatures.

## Producers and reproduction

- Initial callback comparison, clean producer:
  `bc1341ed860711a16ed0755d86d44fd890022963`.
- Default result-materialization comparison: the same before producer versus
  clean `b089b442ccee95e2743e2c28e2c65499fc1ed029`.
- Whole-TensorCircuit/JAX control and portable paired-run coordinator:
  `8c23640dfbd2f41a65d4bcb4ea9f74cdc634858a`.
- The no-callback reporting worker is the `reproduce.py` at the latter two pins;
  the initial callback worker is pinned separately. The original paired
  coordinator is retained inside its capsule; `run_paired.py` provides portable
  paths for the same measured protocol.
- Python 3.12.14, Torch 2.14.1, JAX/jaxlib 0.10.2, TensorCircuit 1.10.0,
  Apple M5/macOS, CPU exact complex128/float64. Process receipts retain versions.
- Both FQ checkouts use the identical loadable native extension SHA-256
  `a7b5f80e48a1c387e6020d8b8f0e8d6951659ab4079a456963c3b5a32d8f551c`.
  These FQ routes use Torch backprop/functional VJPs, **not** a measured native
  whole-circuit adjoint. Availability alone is not execution evidence.

Use an environment with the repository and its optional JAX/TensorCircuit
benchmark dependencies. From the repository root:

```bash
PYTHONPATH=. python artifacts/development/torch_training_cpu_20261103/verify.py
PYTHONPATH=. python artifacts/development/torch_training_cpu_20261103/reproduce.py --output /tmp/training-callback-new
PYTHONPATH=. python artifacts/development/torch_training_cpu_20261103/run_paired.py --before /path/to/clean-before --after /path/to/clean-after --output /tmp/training-paired-new
PYTHONPATH=. python artifacts/development/torch_training_cpu_20261103/compare_whole_jax.py --output /tmp/training-whole-jax-new
```

The candidate diagnostic deliberately replaces a private promotion gate. It
falsifies the implementation and is not a user-facing adoption recipe. Ordinary
usage requires no private import and currently keeps compilation eager:

```python
import flagquantum as fq
import torch

def build(p):
    return fq.Circuit(1).ry(0, p[0])

model = fq.Module(build, 1, init=torch.tensor([0.25]))
result = fq.train(model, optimizer=torch.optim.SGD(model.parameters(), lr=0.2),
                  objective=lambda value: value.square().mean(), steps=2)
assert result.completed_steps == 2
assert result.last_execution.runtime["training_compilation"]["status"] == "eager"
```

## Frozen protocol and timing scope

The six width/depth/batch/layout cases were reserved in the approved contract
before implementation: (3,2,1,chain), (3,4,8,ring), (5,2,8,ring),
(5,4,64,chain), (7,2,64,ring), (7,4,1,chain). All use seed 20261102,
shared nonlinear parameters, changing data, nonuniform vector-output cotangents,
both SGD and Adam, 60 updates and a checkpoint at update 30. Each case/optimizer/
route has three serial fresh processes, a 120-second deadline and a 2 GiB worker
RSS budget. All 180 processes completed; none was removed or replaced.

`callback.tar.gz` retains 72 FQ processes with per-update losses, gradients,
parameters and consumed hashes. Route order reverses in trial two. Each process
has an isolated empty Inductor filesystem cache. `unobserved.tar.gz` retains
72 FQ processes without callbacks, alternating before/after phase order across
three paired rounds. The checkpoint is saved by the data provider between
updates 30 and 31. `whole_jax.tar.gz` retains 36 whole-training controls using
TensorCircuit plus `jax.jit(value_and_grad + jax.example_libraries.optimizers)`.
This is a separate whole-JAX interface, not an official hybrid Torch interface.
JAX persistent compilation caching is disabled and each process is fresh.

All timers include first capture, updates, host receipt work and the checkpoint
write. Imports/setup, first update, every update interval, steady median, total
training, checkpoint and recovery are recorded separately. Setup includes model/
optimizer construction; recovery includes representative-input priming and a
fresh Module/optimizer. FQ checkpoint binaries stay outside Git. External JAX
uses a serialized JAX optimizer state and does not claim Torch checkpoint
compatibility. RSS is the peak worker process across imports, training and
recovery, not simultaneous aggregate compiler-child memory. Framework thread
pools can differ; OMP/MKL/OpenBLAS are set to one, XLA threads are not constrained.
The external campaign runs after FQ, without cross-framework counterbalancing.

## Retained results

Values are three-process medians in seconds. The default comparison has no
callback; the candidate ratio compares the two callback routes within its
separate campaign. Whole-JAX includes its first compilation and host receipts.
These columns have different interface/metadata costs and are not a universal
framework ranking.

| Case (width/depth/batch) | Optimizer | Default before | Default after | Before/after | Compiled/eager total | Whole-JAX total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3/2/1 | SGD | 0.1356 | 0.1315 | 1.031 | 47.7 | 0.2373 |
| 3/2/1 | Adam | 0.1312 | 0.1183 | 1.109 | 42.7 | 0.2537 |
| 3/4/8 | SGD | 0.1827 | 0.1666 | 1.096 | 42.5 | 0.4649 |
| 3/4/8 | Adam | 0.1825 | 0.1734 | 1.052 | 42.2 | 0.4724 |
| 5/2/8 | SGD | 0.1795 | 0.1679 | 1.069 | 47.6 | 0.4883 |
| 5/2/8 | Adam | 0.1774 | 0.1653 | 1.074 | 49.8 | 0.4757 |
| 5/4/64 | SGD | 0.4081 | 0.3577 | 1.141 | 31.6 | 0.8196 |
| 5/4/64 | Adam | 0.4076 | 0.3563 | 1.144 | 29.4 | 0.8210 |
| 7/2/64 | SGD | 0.3516 | 0.3109 | 1.131 | 27.9 | 0.7277 |
| 7/2/64 | Adam | 0.3471 | 0.3155 | 1.100 | 25.0 | 0.7976 |
| 7/4/1 | SGD | 0.3096 | 0.2824 | 1.096 | 32.5 | 1.6225 |
| 7/4/1 | Adam | 0.3090 | 0.2860 | 1.080 | 27.3 | 1.6640 |

All default before/after losses, final parameters and optimizer slots are exact.
Compiled FQ and whole-JAX maximum parameter discrepancies versus eager FQ are
4.22e-15 and 1.04e-14, below the unchanged 1e-10 numerical tolerance. Every
retained gradient/loss/update is compared by the verifier. All FQ checkpoint
replays validate parameters, losses, state, optimizer slots and consumed hashes.
The verifier independently rebuilds all 4,320 callback consumed hashes from each
route's own recorded pre-update parameters and changing data; all agree.
Across compiled/eager trajectories, 1,848/2,160 hashes differ because their
parameters differ at floating-point rounding scale. No hash is compared with a
relaxed tolerance or rewritten to disguise that difference.

The compiled route is not a whole step: the objective/value/VJP graph and SGD
optimizer graph are separate. Its actual graphs, declines, restart counts,
graph-break reasons, first-call and backend-build costs remain in every receipt.
Adam updates use the original eager optimizer. Candidate steady-state median
ratios vary by case; cold losses are not erased by warm kernel observations.
Whole-JAX has substantially lower steady update costs but also a first-compile
cost and a different interface. Inspect `summary.json` and all raw intervals;
three processes do not establish broad significance or arbitrary-workload wins.

## Remaining acceptance and publication

The private compilation gate stays closed. Default improvement does not prove
compiled default adoption. The unchanged #918 public-JAX/default accessibility
budgets have not been rerun or promoted by this packet. The existing generated
matrix and classifier/Heisenberg protocol remain separate required comparisons.
Native whole-circuit adjoint, matched time-to-a-predeclared-quality target,
invalidation cost and compiled long-job amortization remain outstanding. No
threshold is selected from a favorable case, and neither issue is closed.

`publication.json` binds raw and sanitized capsule-member hashes, deterministic
capsule hashes and full producer file pins. `verify.py` validates them and
regenerates `summary.json`, numerical comparisons and consumed-hash checks.
The capsules include all logs and partial-receipt material. Sanitization uses
`tools.sanitize_public_evidence.sanitize_text`; no checkpoint binary is published.

## Additional checkpoint blocker found by real CUDA CI

[Run 41482999465](https://github.com/flagos-ai/FlagQuantum/actions/runs/41482999465/job/125406746309)
passes the existing 19 GPU tests but fails both new fq.train checkpoint recovery
cases. Frozen features combined with a detached parameter-derived rotation are
classified as constant angles by the existing v1 workload signature. Priming a
fresh Module at its initial parameter values then fails workload admission.
The same CPU reproducer fails on the pre-implementation producer
`97b315ea02863583e88d13a4127be65133485cf6` and current source; its trainable-feature
control restores. `checkpoint-blocker.json`, the reproducer and sanitized GPU
failure log retain this result. The failing GPU tests are kept intact.

The [review-only checkpoint proposal](checkpoint_proposal.md) describes a
versioned v2 workload identity based on canonical IR and existing bindings while
preserving strict legacy v1 reading. The accompanying identity fake passes ten
checks, including frozen-feature replay and refusals for changed topology,
gradient cuts and static constants. This fake changes no production loader,
serialized default or public schema and does not certify migration/RNG behavior.
The previously approved #936 contract preserved checkpoint semantics; this
protected extension needs separate approval. PR #1022 stays draft and unmerged.
