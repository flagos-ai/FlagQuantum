# `fq.Module` And `ExecutionResult`

`fq.Module` is the stable PyTorch-native owner of quantum parameters, circuit
construction, observable selection, runtime policy, and deployment binding.
Its `forward()` method returns a tensor, so optimizers and ordinary PyTorch
training loops work without adapters. `execute()` returns `fq.ExecutionResult`
with stable value, state, samples, plan, accuracy, metrics, provenance, runtime,
and compatibility fields.

```python
import flagquantum as fq
import torch

def circuit(parameters, inputs=None):
    program = fq.Circuit(2, device=parameters.device)
    return program.ry(0, parameters[0]).cx(0, 1).ry(1, parameters[1])

model = fq.Module(circuit, 2, policy=fq.RuntimePolicy(observable_wires=(1,)))
optimizer = torch.optim.Adam(model.parameters())
loss = model().sum()
loss.backward()
optimizer.step()
result = model.execute()
```

Request several per-wire Z features in one execution by selecting multiple
observable wires. The result keeps the observable axis instead of reducing it:

```python
features = fq.Module(
    circuit,
    2,
    policy=fq.RuntimePolicy(observable="z", observable_wires=(0, 1)),
)()
assert features.shape == (1, 2)
```

The leading dimension is the circuit batch dimension, so unbatched circuits
return `(1, observable_count)` and batched circuits return
`(batch_size, observable_count)`. Use
`observable="z_sum"` when the observable axis should be summed. Vector-valued Z
execution is supported by the local PyTorch statevector, MPS, and
tensor-network fast paths. JAX and distributed statevector execution remain
single-observable and fail closed when fallback is disabled.

`forward()` uses a tensor-only local training path: it reuses the compiled
circuit builder and skips `ExecutionResult`, IR snapshot, topology hash,
provenance, and runtime-summary construction. Call `execute()` whenever those
audit fields are required.

Use `mode="distributed_statevector"` under an initialized process group to
retain the same module and result surface while forward and backward use the
native sharded statevector runtime. PyTorch is always the stable default. A
non-PyTorch backend request either fails explicitly or records the selected
PyTorch compatibility fallback in the result.

Construct `fq.Module` directly. Legacy layer adapters are intentionally not
part of the stable API; integrations must provide a circuit builder and an
explicit runtime policy.

Module parameters, policy, and deployment binding participate in
`state_dict()` save/load. The circuit builder remains application code and must
be supplied when reconstructing the module, matching normal PyTorch module
class construction.

Named parameter groups remove positional-index bookkeeping for larger circuits:

```python
def named_circuit(parameters):
    return (fq.Circuit(2)
            .ry(0, parameters["encoder"][0])
            .rx(1, parameters["readout"]))

model = fq.Module(
    named_circuit,
    parameters={"encoder": (4,), "readout": ()},
    init={"encoder": "uniform", "readout": 0.1},
    seed=42,
)
```

`init="uniform"` samples angles from `[0, 2π)`, while `init="normal"`
samples from a zero-mean normal distribution with standard deviation `0.01`.
Both work for flat and named parameters; `seed` uses a module-local generator
and does not reset PyTorch's global random state.

A symbolic circuit can be used directly. Its `fq.Parameter` names are inferred,
registered as scalar PyTorch parameters, and bound automatically on execution:

```python
theta = fq.Parameter("theta")
template = fq.Circuit(1).ry(0, theta)
model = fq.Module(template, init={"theta": 0.2})
```

Gate requirements are discoverable without reading implementation code:

```python
import flagquantum.operators as fqo

info = fqo.gate_info("u3")
print(info.n_wires)          # 1
print(info.parameters)       # ("theta", "phi", "lbd")
print(info.parameter_shapes) # each parameter is scalar: ()
```

`fq.train()` is silent by default. Set one positive interval to enable concise
terminal progress without a redundant `verbose` flag:

```python
result = fq.train(
    model,
    optimizer=optimizer,
    objective=lambda value: value.mean(),
    steps=100,
    log_interval=10,
    callback=lambda step, loss, execution: record(step, loss),
)
```

The first and last steps are always printed when logging is enabled. The
callback runs after every optimizer step and is independent of `log_interval`.
