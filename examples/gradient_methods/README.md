# Gradient methods

`fq.gradient(fn, parameters, loss=...)` returns both the derivative of a program
and the method that produced it. This example runs every method one circuit can
use and prints what each reported.

```bash
python examples/gradient_methods/run.py
```

## What it shows

| Printed method | Exact | Why the run reported it |
| --- | --- | --- |
| `autograd` | yes | The program builds an autograd graph, so `method="auto"` measured it and used the graph. |
| `autograd` in `mps` mode | yes | The storage mode changed; the derivative did not. |
| `parameter_shift` | yes | Requested explicitly. The opcode declares the shift rule, so no displacement is needed. |
| `finite_difference` | no | Requested explicitly. The reported `step` is the displacement actually used. |
| `spsa` | no | Requested explicitly. `directions` and `generator` belong to this method alone. |
| `finite_difference` | no | Not requested. The program scored itself under `torch.no_grad()`, so `"auto"` found no graph and fell back. |

## What to take from it

- Read `result.method`; do not assume the route that was requested. `"auto"`
  measures the program and reports what it found.
- `result.exact` is `False` for every approximate route, and `result.step` is the
  displacement that route used. An exact route never reports a step.
- `result.gradient` is detached, so it can be handed to an optimizer or written
  to a report without extending the autograd tape.

`method="adjoint"` is refused: FlagQuantum has no standalone adjoint entry point,
because the reversible adjoint sweep is the backward pass behind PyTorch
autograd and no result reports whether backward used it. Use `"autograd"` and
read the execution mode instead.

The full reference is the
[differentiation section](../../docs/reference/API.md#differentiate-a-program)
of the API reference.
