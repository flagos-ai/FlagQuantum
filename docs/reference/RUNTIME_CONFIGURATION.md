# Runtime configuration

`RuntimeConfig` is FlagQuantum's immutable execution policy. It records backend,
device, real/complex precision, JAX precision, matrix multiplication policy and
drawing style. `Circuit` captures a configuration when constructed and embeds a
versioned manifest in IR and execution plans so distributed workers can rebuild
the same policy with `RuntimeConfig.from_manifest`.

Use an explicit configuration for long-lived or distributed work:

```python
import flagquantum as fq

config = fq.RuntimeConfig(device="cuda")
circuit = fq.Circuit(4, config=config)
```

Use `runtime_config` for ergonomic temporary overrides. Overrides are backed by
`ContextVar`: nested contexts restore exactly, asyncio tasks inherit a snapshot,
threads start with their own default context, and processes reconstruct from the
manifest rather than inheriting mutable state.

```python
with fq.runtime_config(complex_dtype="complex128"):
    circuit = fq.Circuit(2)  # captures complex128
```

Matrix and kernel caches include immutable device/precision policy in their
keys. The compatibility APIs `set_dtype`, `set_backend`,
`set_global_precision`, and `use_style` now update only the current context.
They remain supported through 0.3.x; new execution code should pass
`RuntimeConfig` explicitly. Their process-global interpretation is removed and
will not be restored.

## Precision flow

Each execution resolves complex precision once. `complex64` implies float32
parameters and real-valued components; `complex128` implies float64. The
resolved value controls planning bytes, state allocation, parameter tensors,
gate matrices, and distributed development simulation. These stages must not
independently consult process defaults.

An already-created distributed device is accepted only when its precision
matches the execution request. Conflicting `dtype` and `precision` inputs, or a
device whose state precision differs from the requested execution, fail before
gate application. Double-Single is a separate explicit representation: its
high and low words remain float32 on the selected device and are never inferred
from a native `complex128` label.
