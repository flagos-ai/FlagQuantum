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

### Read a dynamic circuit after feedback

Use the existing dynamic builder to declare intermediate measurements and
feedback. For a local CPU PyTorch run with one circuit batch, request final
Z-basis samples, counts, or both in full logical qubit order.

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2).h(0).measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)
result = fq.run(
    circuit, outputs=(fq.samples(name="final"), fq.counts()), shots=100
)
register = result.measurement("mid_circuit_measurement").value
final = result.measurement("final").value
counts = result.counts[0]
```

The register contains the classical slots after execution. The example writes
one slot, so its shape is `(100, 1)`. The final sample tensor has shape
`(100, 2)`, and the counts are a histogram of those same 100 rows. Leave the
final readout to the executor when using explicit outputs. Programs with a
terminal measurement instruction are outside this workflow.

Run the [complete example](../../examples/local/dynamic_final_outputs.py)
with `python -m examples.local.dynamic_final_outputs`.

### Reuse a Clifford sampler

With the Stim 1.16.0 extra, `circuit.compile_sampler(seed=7)` prepares
one stream for repeated sampling of explicit measurements. Call
`sampler.sample(shots=32)` for bool records or set `bit_packed=True` for
packed rows. To write directly to a fresh local path, call
`sampler.sample_write(shots=32, filepath=path, format="b8")`.
The [prepared sampler example](../../examples/qec/compiled_measurement_sampler.py)
shows repeated writes to a classical slot and verifies the resulting file.

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
