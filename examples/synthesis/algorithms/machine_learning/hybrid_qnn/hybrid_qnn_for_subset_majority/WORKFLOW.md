# Subset-majority hybrid training

Run from the repository root with the ordinary CPU PyTorch dependencies:

```sh
python -m examples.synthesis.algorithms.machine_learning.hybrid_qnn.hybrid_qnn_for_subset_majority.workflow
```

This follows the original Notebook's default: load its pretrained model and
continue for one epoch. All five original tensors are stored in `pretrained.json`
as inspectable numbers, preserving their float32 values without pickle or an
upstream SDK. `provenance.json` records the pinned Notebook, reviewed code cells,
original checkpoint blob/digest and conversion. Only `qlayer.weight` changes its
name to the existing `fq.Module` parameter name.

The data are all 1024 ten-bit strings in the Notebook's seed-1 Python shuffle.
The label is one when bits at indices 1, 3, 4, 6, 7 and 9 contain at least four
ones. Rows 0..339 train the model; rows 340..851 form the disjoint held-out set.
The remaining 172 rows are unused, as in the source. Batches contain 64 rows and
include the final incomplete batch.

The network preserves Flatten, Linear(10,4), Tanh, the two-qubit quantum layer,
Linear(2,1) and ReLU. Dense angle encoding uses RX(pi*x[2k]) then RZ(pi*x[2k+1]);
the ansatz uses two RY angles and one RZZ angle. Readout returns the two logical
probabilities of one, `(1 - <Z>)/2`, in source order. The existing `fq.Module`
keeps live tensors from the classical encoder through quantum execution to the
classical head. SGD uses learning rate 0.05 and MSE, updating all five parameter
tensors. The final RZZ changes the complete state but commutes with both Z
readouts, so its gradient is zero for this source architecture. It is retained.

This local migration explicitly substitutes exact complex128 statevector
expectations and float64 classical tensors for the Notebook's 4096-shot QLayer
and float32 tensors. It does not reproduce shot noise or claim identical sampled
optimization trajectories. It requires no Classiq installation, account, remote
execution or GPU. The quantum program and its independent Circuit comparison
remain runnable with the commands in [README.md](README.md); they check the
complete amplitudes and absolute phase, separately from training.

The default CPU replay produced:

| Held-out metric (512 rows) | Before continuation | After one epoch |
| --- | ---: | ---: |
| MSE | 0.0792270230616618 | 0.07689654839921231 |
| Rounded-label accuracy | 0.96484375 | 0.9609375 |

Accuracy follows the source's `round(prediction)` rule, including ties; it does
not clamp outputs or substitute a sigmoid. Loss is measured over every held-out
row. Printed epoch losses are sample-weighted means over all training batches,
rather than the original Notebook's last-batch printout. A smaller MSE does not
guarantee a higher classification accuracy.

To exercise all training steps without loading a pretrained model:

```sh
python -m examples.synthesis.algorithms.machine_learning.hybrid_qnn.hybrid_qnn_for_subset_majority.workflow --from-scratch --epochs 30
```

With initialization seed 1 and batch-shuffle seed 0, this run reduced held-out
MSE from 0.2864316891806135 to 0.22128458081382438; rounded accuracy remained
0.662109375. That equals the held-out constant-zero baseline, so this run proves
training execution and parameter updates, not successful learning from scratch
or a convergence claim. The pretrained default is reported separately.

Verification:

```sh
python -m pytest tests/integration/test_synthesis_native_qnn_training.py tests/integration/test_synthesis_native_qml_encoders.py -q
```

Tests reproduce the exact data split and labels, compare every complex amplitude
of the trainable circuit with the native synthesized program, and compare the
full network's outputs, input gradients, parameter gradients, SGD trajectory and
held-out metrics with an independent Bloch-vector formula. Finite differences
check quantum input/angle gradients. Both pretrained continuation and 30 epochs
from scratch execute. Evidence applies only to this exact local CPU workflow;
there is no throughput, hardware, distributed or release claim.
