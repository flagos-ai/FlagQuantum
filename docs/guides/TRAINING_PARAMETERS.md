# Training parameters with PyTorch

`fq.Module` owns ordinary `torch.nn.Parameter` objects. Use `init` for starting
values, `requires_grad` for differentiation, and optimizer parameter groups for
updates. These are separate choices. Run the complete local example with:

```bash
python -m examples.local.train_parameters
```

## Starting values and ownership

```python
import torch
import flagquantum as fq

def build(parameters):
    return fq.Circuit(1).ry(0, parameters["encoder"]).rx(0, parameters["readout"])

model = fq.Module(
    build,
    parameters={"encoder": (), "readout": ()},
    init={"encoder": 0.4, "readout": 0.7},
    dtype=torch.float64,
)
```

The module copies initializer values into its own registered parameters. Passing
a Tensor with `requires_grad=False` as an initializer does not freeze the new
parameter. For external differentiable values, the existing
`model(parameters=...)` override is per call and does not replace owned state.

For flat weights, use `n_parameters` and a Tensor initializer. `init="zeros"`,
`"normal"`, and `"uniform"` are also supported; use `seed` for reproducible random
initialization. Named groups can select these strategies individually. Omitted
named initializers use their specified Tensor values or zero values for declared
shapes. Initial values are reshaped to the declared shape and converted to the
requested parameter dtype/device; use an explicit dtype when precision matters.

Zero initialization can be stationary. For `RY(theta)` with Z readout,
`prediction = cos(theta)`, so `prediction.square()` has zero derivative at
`theta=0`. The nonzero values above avoid this particular stationary point.
Random or nonzero initialization is not a general guarantee against vanishing
gradients. Inspect gradients for the actual circuit and objective.

## Freeze before training, then unfreeze

```python
encoder = model.parameter_groups["encoder"]
encoder.requires_grad_(False)
encoder.grad = None
optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)

optimizer.zero_grad(set_to_none=True)
loss = model().square().mean()
loss.backward()
optimizer.step()

encoder.requires_grad_(True)
encoder.grad = None
# The next forward/backward/step also updates encoder.
```

Keeping all parameters in the optimizer makes later unfreezing straightforward.
With standard PyTorch SGD/Adam/AdamW, a parameter whose gradient is `None` is
skipped, including momentum and weight decay. Freeze between complete optimizer
steps and clear any old gradient: changing `requires_grad` alone does not remove
an already computed `.grad`, which an optimizer can still consume. Continue
using `zero_grad(set_to_none=True)` for subsequent steps.

Optimizer history remains caller-owned. Freezing does not reset existing Adam
moments or SGD momentum; unfreezing retains that history. If a new training phase
needs fresh history, explicitly reset the selected optimizer state or create a
new optimizer as a deliberate training decision.

When all parameters and inputs are frozen, run inference without backward: the
result has no differentiable training path.

If the optimizer was constructed from only the trainable parameters, unfreezing
also requires adding the parameter with `optimizer.add_param_group(...)` or
rebuilding the optimizer with the intended state handling. Changing
`requires_grad` alone does not add optimizer membership.

Use `model.named_parameters()` for a uniform view of flat and named parameters.
`model.parameter_groups` exposes named model tensors; it is distinct from
`optimizer.param_groups`, which contains update settings and parameter lists.
Use separate named parameters for independently frozen regions of a weight
tensor. Zeroing selected gradients alone does not protect tensor elements from
optimizer momentum or weight decay. Replace registered Parameter objects only
with deliberate optimizer rebinding; existing optimizers retain old references.

## Resume weights, freezing and optimizer history

Qualified checkpoints now write `flagquantum.training_state.v3`. They restore
registered parameter values and `requires_grad` flags in place, clear stale
registered gradients to `None`, and resume optimizer moments. Reconstruct the
same named optimizer groups in the same order; a different parameter order is
refused even when group sizes match. The optimizer may include frozen weights
or omit trainable ones, provided that membership agrees on both sides.

```python
from pathlib import Path

# Save after a complete optimizer step; the source has executed at least once.
encoder = model.parameter_groups["encoder"]
encoder.requires_grad_(False)
encoder.grad = None
path = model.save_checkpoint(Path("training.pt"), optimizer=optimizer, seed=11)
restored = fq.Module(
    build,
    parameters={"encoder": (), "readout": ()},
    init={"encoder": 0.4, "readout": 0.7},
    dtype=torch.float64,
)
restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.03)
restored.load_checkpoint(path, optimizer=restored_optimizer)
assert not restored.parameter_groups["encoder"].requires_grad
assert all(parameter.grad is None for parameter in restored.parameters())
```

V3 qualifies certified captured callable builders and ordinary Circuit templates
with named symbolic parameters, default construction initial state and no Tensor
literals in source IR. Callable builders requiring inputs retain their existing
representative-input priming requirement. The workload identity includes static
values, topology and gradient cuts, but excludes bound values and freeze flags.
A changed source template or captured constant refuses saving until the source
is rebuilt and executed. Arbitrary classical host code is not fingerprinted by
the existing single-quantum-owner checkpoint boundary.

Unqualified all-trainable workloads keep their historical v1/v2 format and
limitations. Composite models with custom PyTorch extra-state methods, including
the maintained hybrid models' deployment metadata, use this historical path.
They do not acquire v3 rollback or freeze-state guarantees. Unqualified workloads
containing frozen registered parameters refuse save. V3 refuses registered
parameter aliases, duplicate or unregistered optimizer
references, custom state-loading hooks/methods, custom optimizer implementations,
shared tensor storage and custom/non-strided Parameter storage. Reusing one
parameter in multiple gates remains supported. These checks run before target
mutation. Failed supported restores roll back values, flags, original gradient
objects, optimizer state, precision and RNG.

Checkpoints represent complete optimizer-step boundaries. Save and load between
steps; accumulated gradients and live autograd graphs are not restored. GPU,
distributed and cross-Torch-version support are not expanded by this change.
The exact contract is in the
[approved v3 proposal](../development/TRAINING_PARAMETER_STATE_PROPOSAL.md).

### Historical files

The reader still accepts v1/v2 under their original workload algorithms. They
lack freeze metadata, so reapply `requires_grad` explicitly before restoring
such files; their historical workload limitations remain. To migrate, restore
the old file, reapply the freeze selection, execute the qualified workload and
save a new v3 checkpoint. Never relabel an old payload or reuse its digest as v3.
Older readers reject v3 files as an unsupported version.
