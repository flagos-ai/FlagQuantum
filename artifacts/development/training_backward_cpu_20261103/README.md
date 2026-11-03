# CPU training backward for batched disjoint rotations

The twentieth backward of frozen case 3 with Adam, 64 examples and
complex128 state was profiled separately from timing. The largest self-time
operator group was the four batched matrix multiplications that form the
dense gate gradients, with shapes [64,16,2] and [64,2,16]. They used 0.695 ms
of self CPU time. The baseline backward also spent time reducing the
broadcast products used to form the Kronecker matrix. After the change,
the dense matrix gradient group is absent from the profile. Total
instrumented backward self CPU time was 3.648 ms before and 2.503 ms after.
Those are single profiler samples.

The local statevector executor applies four or more disjoint one-wire
matrices individually using the existing matrix apply path for large
CPU batches with autograd enabled. The threshold of 64 follows the
measured large-batch cases. An initial broader trial regressed at batch
eight. The rule also requires the dense matrix per example to contain at
least as many elements as the state. This limits the change to the regime
where avoiding the matrix construction can offset the extra state passes.
Inference, single-state, small-batch and multi-wire regions retain their
previous execution path.

## Paired comparison

The campaign reused all six shapes, both optimizers, the frozen feature
schedule, seed 20261102, 60 updates and the previously recorded objective
thresholds. All 72 workers passed and all pairs reached the same threshold
at the same observation. The harness loads the old apply function from
the pinned baseline Git object in the baseline process. Both legs include
the same current builder code, including conditional FX regeneration.
The environment used Apple M5 CPU, Python 3.12.14 and Torch 2.14.1 with one
Torch thread. Framework loading, setup and time through the first observed
threshold crossing are included. The harness's baseline code injection and
interpreter launch are excluded. The target is the frozen training objective.

| Batch-64 case | Optimizer | Cold time to target, before → after | Steady update, before → after |
| --- | --- | ---: | ---: |
| Width 5, depth 4 | SGD | 1184.21 → 1137.58 ms | 4.999 → 4.093 ms |
| Width 5, depth 4 | Adam | 1168.90 → 1122.49 ms | 5.017 → 4.110 ms |
| Width 7, depth 2 | SGD | 1130.29 → 1112.51 ms | 4.207 → 3.975 ms |
| Width 7, depth 2 | Adam | 1076.49 → 1068.32 ms | 4.216 → 4.000 ms |

Each value is a three-process median. Batch-64 cold time fell 0.8–4.0%
and steady medians fell 5.1–18.1%. Small-batch medians varied between legs
with the same numerical dispatch. Results are scoped to this CPU and
these frozen workloads.

The maximum paired errors were 5.55e-16 in loss, 5.83e-16 in gradient,
and 3.32e-15 in parameters. Each process reloaded a step-30 checkpoint
into a fresh Module and verified the remaining losses, state, optimizer
slots and final consumed IR hash against its uninterrupted trajectory.
All recovery parameter errors were zero. Cross-route numeric IR hashes
can differ when training trajectories round differently. The independent
state and cotangent tests compare the candidate against sequential
execution for both complex64 and complex128, including gradients of the
initial state and gate angles. The focused test suite passed 324 tests.

## Reproduce

```bash
D=artifacts/development/training_backward_cpu_20261103
PYTHONPATH=. python "$D/profile_backward.py" baseline
PYTHONPATH=. python "$D/profile_backward.py" candidate
PYTHONPATH=. python "$D/run.py" --output /tmp/fq936-backward.json.gz
python "$D/run.py" --verify --output /tmp/fq936-backward.json.gz
```

## Remaining acceptance

- **#936** gains a faster ordinary eager CPU route for the measured large
  batches. The existing compiled candidate still needs a default CPU
  end-to-end benefit and a certified compiled Adam update. The earlier
  same-batch comparison measured compilation and shape-change penalties.
- **#918** still needs matched comparisons with the strongest supported
  public training routes on realistic tasks, comparable batch execution
  and demonstrated default accessibility. This experiment extends the
  frozen shape matrix and does not replace the original task comparison.
