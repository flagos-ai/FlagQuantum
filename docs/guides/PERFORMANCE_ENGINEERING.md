# Continuous performance and memory engineering

FlagQuantum measures performance as a correctness-gated architectural property.
The catalog in `benchmarks/performance_catalog.py` covers micro, kernel, circuit,
training-step, communication, compile-time, startup and repeated-step memory
layers. PyTorch and JAX results remain separate, as do local fast-path latency,
distributed throughput and rank-sharded capacity behavior.

Every performance artifact records warmup and steady-state samples, robust and
ordinary variance, initialization/teardown, allocated/reserved peak and slope,
useful-work/idle/communication fractions, heartbeat gaps, completed work units,
correctness and hardware-specific thresholds. Median absolute deviation controls
short-kernel outliers while the ordinary coefficient remains visible.

Hardware jobs publish run-scoped artifacts through CI. A reviewed baseline may
be supplied to `tools/evaluate_performance_artifact.py`; absent baselines never
silently become release evidence. Only current, public claims belong under
`benchmarks/results/`.

The optimization backlog covers gate fusion, layouts, allocation reuse,
vectorization, compile/JIT caching, communication overlap and vendor/custom
complex kernels. A performance change must attach before/after artifacts and
retain its correctness gate.

## Training through the public interface

Create and reuse one `fq.Module` outside the optimizer loop so static builder
and backend compilation caches survive parameter updates. Its default backend
is PyTorch. When optional JAX is installed, the existing JAX policy exposes
compiled quantum execution through the same PyTorch training interface:

```python
import torch
import flagquantum as fq

def build(theta):
    return fq.Circuit(2).ry(0, theta=theta[0]).cx(0, 1)

model = fq.Module(
    build, n_parameters=1, init=torch.tensor([0.2]),
    policy=fq.RuntimePolicy(execution_options=fq.ExecutionOptions(
        backend="jax", mode="statevector", allow_backend_fallback=False,
    )),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
for _ in range(10):
    optimizer.zero_grad(set_to_none=True)
    loss = model().square().mean()
    loss.backward()
    optimizer.step()
```

For a runnable Torch/JAX comparison, use
`python examples/single_machine_quantum_ai/04_jax_kernel_torch_layer.py --steps 2 --compare-torch`.
Record first-call compilation separately from repeated training and include it
when judging time to a trained model. JAX is not automatically selected and is
not always faster. Repeated small static circuits and short one-off jobs have
different compilation break-even points.

Framework comparisons should include the applicable public Module routes and
verify the installation, optional backend, native-extension availability,
precision and derivative method. An internal kernel timing is a diagnostic,
not a substitute for the user's Module training cost. Keep batching equally
supported and explicit; do not imply the serial-example comparison is each
framework's best throughput configuration. The
[matched training protocol](../development/TRAINING_COMPARISON_PROTOCOL.md)
provides frozen CPU, JAX and installed-native diagnostics. Outstanding public
training-path performance gaps are tracked in
[#918](https://github.com/flagos-ai/FlagQuantum/issues/918).

The [Module execution implementation plan](../development/MODULE_TRAINING_EXECUTION_PLAN.md)
defines the ordered work and its executable circuit-family matrix. The JAX
executor respects current Torch derivative demand: use `torch.no_grad()` or
`torch.inference_mode()` for validation and inference to avoid computing unused
derivatives. `model.eval()` alone retains autograd semantics. Different derivative
demands can incur separate first-call JIT costs, so include those in end-to-end
training measurements.

Implementation histories, failed experiments, and superseded engineering
measurements belong in the external evidence archive. This guide retains only
the practices required to produce reviewable current evidence.
