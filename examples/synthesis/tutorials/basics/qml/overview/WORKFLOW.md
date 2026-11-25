# Source workflow mapping and acceptance

Pinned source: `tutorials/basic_tutorials/qml_with_classiq_guide/qml_with_classiq_guide.ipynb`
at `b69c4893d8264d1e59ce0909df2bd88efed25e00`. Source identity and all 20 reviewed
code-cell hashes are recorded in `provenance.json`.

| Source cells | Native workflow |
| --- | --- |
| 8, 9 | Existing local dependencies; retain Hamiltonian 0.5 I + 0.5 Z - X. |
| 12, 14 | Native U ansatz, zero initial angles, actual quantum cost and existing native optimizer with at most 200 iterations. |
| 17, 19 | Plot actual energy evaluations; report final energy, angles and convergence. |
| 35 | Interpret the source's printed NOT batch as float64 pi/zero inputs and labels 1/0. |
| 40, 42, 44, 46 | Reuse frozen native RX encoding/mixing helpers, named output and resource report. |
| 50, 54 | Actual P0 readout; explicit quantum parameter-shift weight derivative replaces the source's QLayer autograd boundary. |
| 58, 62, 65, 67, 68 | PyTorch SGD, learning rate 1, full-batch MAE, initial weight 3, five epochs and actual update trace. |
| 70, 71 | Final predictions and accuracy with absolute tolerance 1e-2. |

Exact complex128 local statevector execution replaces sampled remote objectives.
VQE reuses public NelderMeadOptimizer; no identical remote optimizer trajectory
is claimed. The source's stored energy can differ from the exact eigenvalue
because its objective was sampled. Store actual local costs and convergence.

The NOT example computes each derivative from two shifted native executions,
multiplies it by MAE's sign derivative (zero at equality), averages both examples
and assigns that gradient to the PyTorch optimizer-owned parameter. Five epochs
mean exactly five full-batch updates. No gradient is inferred by differentiating
through static float synthesis, and no general differentiable QLayer is exposed.
Source float32/shot trajectories and exact float64 local trajectories can differ.

Acceptance compares complete absolute complex states and clean work against
independent Circuit matrices, both P0 and parameter-shift derivatives against
independent Circuit autograd, and every one of the five SGD updates. VQE uses
an independent 2x2 Hamiltonian eigensolution as its acceptance oracle. It also
checks bounded nonconvergence, invalid controls, resource refusals, fresh Notebook
execution, actual figures and CLI, all216+6 replays and default/quality checks.
Final circuit resources are distinct from total quantum objective evaluations.
Classification: `single_device_fast_path`. No distributed, hardware, generic
trainable-qfunc, or quantum-advantage capability is promoted by this example.

## Recorded local acceptance

The complete workflow passed 35 focused integration cases and fresh Jupyter
execution of all four clean code cells plus an independent evidence cell with
11 acceptance scenarios. The actual inline figure was inspected. The CLI also
ran successfully. VQE converged after 113 iterations and 227 quantum objective
evaluations to -0.6180339887498949. Five NOT updates reached weight
3.1371481523084572 and accuracy 1 on the two training examples. Each final bound
circuit uses one qubit, zero clean work bits, one operation and depth one.

All 216 original replay outcomes and six approved-revision outcomes, including
resource rows, remain unchanged. The complete local smoke/unit run reported
19,617 passed, 768 skipped, 26 failed and six errors. The previous baseline had
24 failed and six errors. The added optional NumPy reference was guarded in its
figure test; the fixed dependency check and the unchanged process-cleanup test
both passed isolated replay. The latter initially failed because process
inspection/signaling was denied. The original full-run log is retained; this
is not a passing default tier. The prior 30 failure/error nodes remain unresolved.
No remote CI or release result is claimed.
