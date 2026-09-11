# Publication validation scope

This pre-release supports executable workflows at different maturity levels.
Implemented APIs, passing correctness checks, historical measurements, and
research goals are distinct. The [capability catalog](../generated/CAPABILITIES.md)
and [known limitations](KNOWN_LIMITATIONS.md) remain authoritative for each path;
this review does not promote their maturity levels.

## Evidence reviewed on 2026-09-11

| Area | Implemented and checked | Boundary |
| --- | --- | --- |
| Local learning | Circuit construction, PyTorch Module/autograd, optimizers and local measurements; Python 3.10, 3.11 and 3.12 core CI passes | Gradient support depends on representation and backend; no blanket higher-order-gradient claim |
| GPU and distributed execution | A800 single-GPU gate checks; two-GPU statevector forward, reverse and training; MPS training on 2, 4 and 8 GPUs with checkpoint/resume | Selected correctness workloads, not a performance comparison or capacity certification |
| Cross-node statevector | Two nodes, one A800 per node: forward, gradients and training/resume over NCCL/TCP | The forward probe uses five qubits; no RDMA, throughput, strong-scaling or large-state capacity claim follows |
| Tensor networks | Native slicing and distributed CPU correctness checks | The A800 results above do not certify all tensor-network execution or gradient paths |
| Jiuding and Quafu | Experimental adapters, submission/result contracts and runnable credential-dependent examples | No new live QPU job was submitted in this review; earlier provider evidence remains limited to its recorded task and environment |
| Digital twin | Calibration-conditioned predictions and program/task-bound comparisons | Retrospective replay is not general prospective prediction across circuits, devices or calibration epochs |
| QEC and FTQC | Local three-data-qubit repetition-memory experiments with syndrome, decoder and correction records | General logical operations, threshold claims and real-time QPU feedback remain research goals |

The reviewed CI run passed all 15 jobs. Its coverage job passed 2,455 tests,
skipped 36 and measured **66.10% line coverage** under the repository's existing
`.coveragerc` exclusions. The CPU distributed tier passed 402 tests with 28
skipped; a separate benchmark-contract tier passed 113 with 12 skipped. These
counts are different scopes and must not be added into a unique-test total.
Skipped tests are not evidence of supported hardware or behavior.

The A800 environment was Python 3.12.13, PyTorch 2.13.0+cu130, CUDA 13.0 and
NCCL 2.29.7, on A800-SXM4-80GB devices. Raw logs remain in the maintainer's private
validation archive because they contain infrastructure metadata. This summary
is not a publicly replayable performance artifact or a scalability release gate.

One CI checkpoint-lease fault test failed without captured worker diagnostics.
The harness now exposes worker output and separates normal checkpoint deadlines
from deliberate timeout injection. All 14 fault scenarios passed on Linux, and
the final CI passed; the original failure's root cause was not established.
An additional macOS distributed check hit launcher timeouts and was stopped;
this review does not certify macOS distributed execution.

## Reproduce the relevant checks

Use the dependency installations and tier commands in
[CI](../../.github/workflows/ci.yml) and the
[testing manual](../development/TESTING.md). Hardware scenarios live in
[distributed tests](../../tests/distributed/), including
`statevector_forward_executor.py`, `statevector_reverse_executor.py`,
`statevector_training_runtime.py`, and `test_mps_training_runtime.py`.
The [distributed MPS runbook](../../examples/distributed_mps/RUNBOOK.md)
provides workload and evidence interpretation guidance.

Historical benchmark figures retained in Git history are records of their
original experiments. They do not certify the present
release. Performance comparisons require matched workloads, precision,
measurement methodology and auditable artifacts; FTQC and general predictive
digital-twin claims require their own scientific evidence.
