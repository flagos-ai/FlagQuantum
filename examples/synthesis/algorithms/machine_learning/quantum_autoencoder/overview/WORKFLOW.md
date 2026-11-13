# Source workflow and acceptance

The pinned source is `algorithms/QML/quantum_autoencoder/quantum_autoencoder.ipynb`
at commit `b69c4893d8264d1e59ce0909df2bd88efed25e00`. All 28 code cells are
reviewed and hash-bound in `provenance.json`; source code is not bundled here.

| Original cells | Native behavior |
| --- | --- |
| 6-9 | Local imports, four RY data encodings, two linear-CX/RY layers and final two trash RY rotations. |
| 13-18 | Three domain-wall inputs, two coded/two trash bits, ten weights and a swap test against two zero reference bits. |
| 20-34 | Preserve float32 checkpoint values before float64 adaptation; 4096 shots, L1 loss, SGD lr 0.3, shuffled batches 2/1, one executed epoch. The helper default 40 is not the executed epoch count. |
| 37-42 | RESET both trash bits and adjoint encoder; sample all three decoded outputs. Compare full density matrices, not pure states. |
| 45-47 | All five source anomaly inputs, shuffled locally, P(test=1), strict threshold 0.01. |
| 51-56 | Independent alternative training from the same checkpoint using expected trash population, the same batch/loss/optimizer and one epoch. |

Native array index i is bit i of named q's integer value; trash is bits 2/3.
Physical Circuit wire order is a separate implementation detail verified by
independent full-state tests. Swap cost equals half of one minus the probability
of both trash bits being zero. Dropped swap reference remains entangled retained
environment: seven physical bits, five named output bits, zero clean work bits.

Training explicitly transfers ten parameter-shift gradients into PyTorch SGD.
Each input uses one forward plus twenty shifted quantum executions. Both routes
therefore require 126 executions, plus three decoder and five anomaly executions:
134 total. Default readouts each draw 4096 actual outcomes from local complex128
execution probabilities; no analytical cost replaces quantum execution. Explicit
shots=None is reserved for independent exact-gradient acceptance.

The source leaves loader and measurement seeds unspecified. This workflow fixes
loader seed 7, anomaly order seed 19 and documented per-evaluation seeds.
Checkpoint float32 rounding is preserved before local float64 arithmetic. The
alternative observable is also explicitly estimated with 4096 shots here.
No identical original sampled trajectory or generic differentiable qfunc layer
is claimed. All three decoded modal inputs and both detected anomalies are
results of execution, not assumed labels.

Tests compare two nonzero bindings in full absolute complex state, all ten exact
gradients against Circuit autograd, same-seed sampled forward readouts, every
minibatch gradient and SGD update, all decoder density matrices and all anomaly
scores. They also cover bit order, swap identity, retained environment versus
clean work, channel positivity/trace/mixedness, partial batches, input rejection,
resource refusal, plots, clean Notebook execution and source hashes. Fresh
Jupyter, CLI, frozen 216+6 replay and repository verification are additional
local delivery gates. Classification: `single_device_fast_path`.

## Recorded local acceptance

All 33 focused cases and 49 catalog/dependency cases passed. Fresh Jupyter
executed five clean code cells plus one evidence cell with ten independent
acceptance scenarios; the actual inline figure was visually inspected. The CLI
completed. Every minibatch prediction, gradient and update matched same-seed
independent Circuit sampling, and all ten exact gradients matched Circuit
autograd at two nonzero bindings. The source checkpoint was also extracted from
the pinned original cell and compared bitwise after float32-to-float64 conversion.

Swap-route minibatch L1 losses were 0.0020751953125 and 0.000244140625;
alternative trash-route losses were 0.00537109375 and 0.001220703125. Three RESET
decoded modal outputs reproduced the original domain-wall inputs. The two
anomalies scored 0.497802734375 and 0.509033203125; the three normal inputs
scored 0.00244140625, 0.001708984375 and 0.000244140625. These are fixed-seed
observations, not universal performance guarantees.

The complete workflow used 134 quantum executions, each with 4096 sampled
outcomes. Per-circuit resources were seven physical bits, 62 operations and
depth 34 for swap testing; four bits, 34 operations and depth 14 for encoding;
four bits, 68 operations, depth 29 and two channel operations for decoding.
All paths used zero clean work bits; the swap reference is retained environment.

All 216 original and six approved-revision outcome/resource rows were unchanged.
The complete local smoke/unit tier reported 19,619 passed, 768 skipped, 24 failed
and six errors. Its 30 failure/error nodes matched the prior unresolved baseline
exactly; none were added or removed. This is not a green complete default tier.
All 4103 Python inputs stayed frozen during that run. The initial run with an
injected GIT_WORK_TREE was terminated and reaped; the accepted terminal run
explicitly removed that variable. Raw logs were retained. After terminal
comparison, only catalog count/entry assertions changed from 80 to 81, followed
by targeted regression. No remote CI, release or hardware claim is made.
