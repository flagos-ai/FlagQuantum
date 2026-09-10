# Local workflows

Local execution is FlagQuantum's zero-configuration path. It requires no
provider account, compiler plugin, task scheduler, or network connection.

## Simulate

```python
import flagquantum as fq

circuit = fq.Circuit(2).h(0).cx(0, 1)
result = fq.run(circuit)
state = result.to_statevector()
```

CPU statevector execution is the default. Select one locally controlled GPU
explicitly when needed:

```python
result = fq.run(
    circuit,
    options=fq.ExecutionOptions(device="cuda:0"),
)
```

`ExecutionOptions.device` describes a device controlled by the current
process. The `target` argument is reserved for external execution destinations
such as Jiuding or Quafu.

## Measure

Request the scientific result you need instead of manually inspecting the
statevector:

```python
probabilities = fq.run(
    circuit,
    outputs=fq.probabilities(),
).probabilities

correlation = fq.run(
    circuit,
    outputs=fq.expectation(fq.Z(0) @ fq.Z(1)),
).expectation()

counts = fq.run(
    circuit,
    outputs=fq.counts(),
    shots=1024,
).counts[0]
```

Probabilities and expectations are exact by default. Counts and samples require
an explicit shot count. Local execution preserves its batch dimension, so
`counts` returns one dictionary per batch item; `[0]` selects the default
single-circuit batch.

## Train

An `fq.Module` owns trainable quantum parameters and behaves as a PyTorch
module:

```python
import torch
import flagquantum as fq

def build(parameters):
    return fq.Circuit(1).ry(0, theta=parameters[0])

model = fq.Module(build, n_parameters=1, init=torch.tensor([0.25]))
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
training = fq.train(
    model,
    optimizer=optimizer,
    objective=lambda value: value.mean(),
    steps=8,
)
print(training.final_loss)
```

Use an ordinary PyTorch training loop when the quantum module is part of a
larger classical model. `fq.train` is a concise single-module workflow, not a
replacement for PyTorch.

## Run the maintained paths

```bash
python -m examples.local.simulate
python -m examples.local.measure
python -m examples.local.train
```

Continue with the [single-machine examples](../../examples/single_machine_quantum_ai/README.md)
for explicit simulator selection, larger models, and configurable CPU/GPU runs.
