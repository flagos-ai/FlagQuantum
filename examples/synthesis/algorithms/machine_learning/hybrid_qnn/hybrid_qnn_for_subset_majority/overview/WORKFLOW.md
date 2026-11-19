# Source mapping and acceptance contract

Pinned source: `algorithms/QML/hybrid_qnn/hybrid_qnn_for_subset_majority.ipynb`.
All 17 code-cell hashes and the checkpoint identity are recorded in
`provenance.json`. The native quantum kernel remains frozen in the parent
folder. No Qmod source or importer is used by this workflow.

| Source cells | Native workflow |
| --- | --- |
| 7 | Existing local dependencies; no installation inside the Notebook. |
| 8, 9, 10, 12 | All 1024 ten-bit words, Python shuffle seed 1, subset 1/3/4/6/7/9, strict majority above 3, first 340 for training and next 512 for testing. |
| 13, 14 | Batch size 64, shuffled train/test loaders, retain partial final batch. |
| 19, 20, 21 | Existing native two-qubit RX/RZ encoding, RY-equivalent rotations, RZZ, logical P(1) outputs and 4096 joint samples per evaluation. |
| 24, 27 | Classical 10-to-4 linear/Tanh, three quantum weights, 2-to-1 linear/ReLU, MSE and SGD learning rate 0.05. |
| 29, 34, 35 | Load original checkpoint tensor values, execute exactly one epoch: five batches of 64 plus one batch of 20. The helper's default 20 and the commented 30 are not executed epoch counts. |
| 31, 36 | Round actual sampled scores and report held-out accuracy on all 512 examples. |

The original float32 checkpoint tensors are exported without numerical change
to portable JSON and represented as float64 locally; loading never unpickles an
external artifact. There is no retraining from random initialization. The source
did not fix Torch loader or shot seeds; the native example uses explicit seed 17
and per-evaluation seeds for reproducibility. No identical source trajectory or
accuracy is claimed. Exact complex128 local simulation supplies joint sampling
probabilities; actual 4096-shot samples supply forward and shifted readout values.

Four encoded features need derivatives as well as the three quantum weights.
Encoding uses pi times each feature, so feature shifts are plus/minus 0.5 and
the multiplier is pi/2. Weight shifts are plus/minus pi/2 with multiplier 1/2.
The example explicitly transfers both quantum Jacobians through the classical
Tanh/linear stages. Static float synthesis itself is not differentiated.

Each training example requires one forward plus 14 shifted quantum executions.
Training therefore uses 5100 executions and testing 512: 5612 total, each with
4096 shots. There is no pruning of zero analytical derivatives or replacement of
sampled gradients with classical closed-form expressions. Report per-circuit
resources separately from this total workload.

Acceptance uses two nonzero bindings for complete absolute complex states,
logical bit order, clean work and every quantum Jacobian against an independent
Circuit/autograd reference. It also compares a nontrivial exact end-to-end
classical/quantum gradient chain, every sampled minibatch gradient/update and all
512 predictions against independent Circuit sampling with the same seeds. Test
partial batches, the strict label threshold, checkpoint values, plots, budgets,
invalid inputs, fresh Jupyter/CLI, frozen 216+6 replay and default/quality checks.
Classification: `single_device_fast_path`. No distributed, hardware, release or
quantum-advantage claim is made.

## Recorded local acceptance

All 24 focused cases passed, including a final replay after the optional plot
dependency guard was added. Fresh Jupyter executed all five clean code cells
and one independent evidence cell with six acceptance scenarios. The actual
inline figure was inspected, and the CLI completed. Every sampled minibatch
gradient/update and all 512 predictions matched independent Circuit execution
with the same seeds. The first classical layer changed by a nonzero amount.

The fixed local seed produced 499 correct test labels out of 512 (0.974609375).
The six observed MSE values were 0.0619380362, 0.0610212709, 0.0629699844,
0.0713079137, 0.0780341760 and 0.0449239615; no monotonic loss reduction is
claimed. There were 5612 quantum evaluations, each with 4096 shots. The final
bound circuit used two qubits, zero clean work bits, nine operations and depth
five. These are per-circuit resources, not total training costs.

All 216 original and six approved-revision outcomes and resource rows remained
unchanged. The complete local smoke/unit tier reported 19,617 passed, 768 skipped,
26 failed and six errors, versus 24 failed and six errors previously. The two
added optional-dependency and generated-navigation failures were repaired after
completion; all 49 catalog/dependency tests and the 24 focused cases then passed.
The prior 30 failure/error nodes remain unresolved. The full-run log and repairs
are retained separately; this is not a passing complete default tier or a remote
CI/release result.
