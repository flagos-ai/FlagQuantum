# The gradient method and execution mode matrix is measured, not declared

- **Status:** implemented
- **Owner:** core (`flagquantum/gradients.py`) with verification (`contracts/`, `tools/`)
- **Date:** 2026-10-27
- **Related:** [`fq.gradient` and the reported gradient method](FQ-GRADIENT-API-20261002.md),
  [Gradient parameter frequencies](FQ-GRADIENT-PARAMETER-FREQUENCIES-20261002.md),
  [The batch parameter-shift profile reads the opcode declaration](FQ-GRADIENT-BATCHED-SHIFT-PROFILE-20261020.md),
  [Circuit composition](FQ-CIRCUIT-COMPOSITION-20261002.md)

## Problem

`fq.gradient(program, parameters, loss=None, *, method="auto", step=None, directions=1,
generator=None)` accepts five methods and refuses a sixth by name. The **execution
mode** is not one of its arguments and never has been: the mode belongs to the caller's
loss, because the loss is what calls `fq.run`. So the availability of a derivative is a
property of a *pair* -- a method chosen at the `fq.gradient` call and a mode chosen
inside the loss -- and there are thirty such pairs per program shape, sixty once a noise
model is attached.

Nothing in the repository stated which pairs produce a derivative, which refuse, and
why. Three things followed from that.

1. **A user choosing a method had no source to read.** `GradientResult.exact` reports
   whether the method that ran was exact, but the availability question -- "can I get a
   derivative at all if my loss runs in `mps`?" -- could only be answered by trying it.
2. **`examples/gradient_methods/run.py` shows the methods under one mode.** It had to:
   there was no place to show the rest. A reader could reasonably infer that the other
   modes are equivalent, which is a guess rather than a fact, and the guess is wrong for
   `stabilizer`.
3. **The scheduling plan proposed the wrong matrix.** `N3-9` in
   `flagquantum-pennylane-alignment-pr-plan.md` sketched method rows
   (`parameter_shift`, `adjoint`, `finite_difference`, `spsa`) against mode columns
   (`statevector`, `mps`, `tensor_network`, `distributed_statevector`,
   `noisy_density_matrix`). Three of those five columns are not dimensions of this
   problem at all, and all three errors were in the direction of over-promising. Each is
   falsified by measurement below. That is the strongest available argument for the
   change: a hand-written matrix of this shape had already been written once, and it was
   wrong in three ways.

## Decision

Write the matrix as a contract and re-measure it in a gate, so that neither half can
drift from the implementation without a failure.

- `contracts/gradient-methods-contract.toml`: the reference program, the two
  vocabularies, `[taxonomy]`, `[rules]`, `[reference]`, `[resolution]`, `[accuracy]`,
  `[cost]`, then `[[cells]]` (30), `[[noise_cells]]` (30), `[noise]`, `[[refusals]]`
  (10), `[[reframings]]` (1, with two measured instances), and `[verification]`.
- `tools/check_gradient_methods_contract.py`: rebuilds the reference program, runs all
  sixty cells, and compares each measured verdict, reported method, and exactness with
  the recorded one. It also reads the two vocabularies out of the implementation, checks
  every refusal phrase against the file it names, re-counts loss evaluations, reproduces
  all three `auto` resolutions, and measures both halves of the recorded reframing.
- `tests/unit/test_gradient_methods_contract.py`: the file's internal consistency and
  the refusal vocabulary, plus a mutation per clause requiring the gate to name it.
- `.github/workflows/ci.yml` and `tools/pre_push.py`: the gate is invoked by both.

### The reference program

One program and one parameter vector, so that "a cell serves" means the same thing
everywhere in the file:

```python
PARAMETERS = torch.tensor([0.3, 0.7, -0.4], dtype=torch.float64)

def build_circuit(parameters):
    return (fq.Circuit(2, dtype=torch.complex128)
            .ry(0, theta=parameters[0]).rx(0, theta=parameters[1])
            .ry(1, theta=parameters[2]).cx(1, 0))

def loss(circuit):
    return fq.run(circuit, options=fq.ExecutionOptions(mode="statevector"),
                  outputs=fq.expectation(fq.Z(0))).expectations[0]
```

`fq.gradient(build_circuit, PARAMETERS, loss, method="parameter_shift")` returns

```text
[-2.081840281210e-01, -5.668620735708e-01, 2.845408368732e-01]
```

for six loss evaluations. That vector is the reference every deviation in `[accuracy]`
is measured against.

### The unnoisy matrix: 25 of 30 cells serve

Measuring all thirty pairs gives one refusing mode and one refusing method.

- **`stabilizer` refuses all five methods.** `CapabilityError: mode='stabilizer' samples
  measurement outcomes; it cannot serve measurement kind(s) expectation_ps`. It is not a
  gradient limitation; the observable this program asks for cannot be estimated from
  samples, so no method could answer it.
- The other five modes (`auto`, `density_matrix`, `mps`, `statevector`,
  `tensor_network`) serve all five methods.

Per cell, the reported method and exactness are uniform within a method row:

| method | reported method | `exact` | step | loss evaluations |
| --- | --- | --- | --- | --- |
| `auto` | `autograd` | `true` | `None` | 1 |
| `autograd` | `autograd` | `true` | `None` | 1 |
| `parameter_shift` | `parameter_shift` | `true` | `None` | 6 |
| `finite_difference` | `finite_difference` | `false` | `6.055454452393343e-06` | 7 |
| `spsa` | `spsa` | `false` | `6.055454452393343e-06` | 9 |

The deviations from the reference vector are `1.110e-16` for `autograd` in every mode
except `density_matrix`, where it is `2.220e-16`; `0.0` for `parameter_shift` in `auto`
and `statevector` and `1.110e-16` in the other three; `8.343e-12` for
`finite_difference` in `tensor_network` through `3.749e-11` in `mps`. The step is
`math.pow(eps, 1/3)` of the loss dtype: `6.055454452393343e-06` for float64 and
`0.00492156660115185` for float32.

### Falsification 1: `adjoint` is not a per-mode row

The plan predicted an `adjoint` row whose availability varied by mode. It does not reach
a mode at all. `method="adjoint"` is refused after the parameter tensor is read and
before any program is touched, for every mode:

```text
CapabilityError: FlagQuantum has no standalone adjoint gradient: the reversible adjoint
sweep is reachable only as the backward pass behind PyTorch autograd, and no result
reports whether backward used adjoint replay. Use method='autograd' and read the
execution mode, or method='parameter_shift'.
```

So `adjoint` is recorded in `refused_methods` and as a refusal row, not as five cells. A
reader who looks for a distributed or noisy adjoint row finds
`no_standalone_adjoint` instead of silence.

### Falsification 2: the five distributed mode names are unreachable

The plan predicted a `distributed_statevector` column. `ExecutionOptions` does not accept
that name, or any of `distributed_mps`, `distributed_tensor_network`, `jax_sharded_mps`,
`jax_sharded_tensor_network`:

```text
ValidationError: mode must be one of: auto, density_matrix, mps, stabilizer,
statevector, tensor_network
```

The runtime's own mode tables know the five names, but the option layer refuses them
before any gradient code runs, so no user-supplied loss can select one. They are recorded
in `unreachable_modes`, and the gate asserts each still raises `ValidationError` from
`ExecutionOptions`; if one becomes reachable, the gate fails until the matrix gains its
column.

### Falsification 3: noise is a `fq.run` argument, not a mode

The plan predicted a `noisy_density_matrix` column. There is no such mode. Noise is a
`noise_model=` argument to `fq.run`; `ExecutionOptions` has no noise field, and
`NoiseModel` is not a root export at all. A noisy program is therefore a *different
program*, not a different column, and the file carries it as a second axis of thirty
cells measured under `NoiseModel().add("cx", depolarizing_channel(0.05))`.

What that axis measures is that **noise narrows the reachable mode set to two**:

| axis | serves | refuses |
| --- | --- | --- |
| unnoisy | 25 | 5 |
| noisy | 10 | 20 |

Only `auto` and `density_matrix` serve, and only for the reason that a noisy program must
be executed as a density matrix. The other four modes are refused with
`ValidationError: stable noisy execution supports mode='auto' or mode='density_matrix'`.
`stabilizer` still refuses with its own capability error, not with that one. The reported
method is unchanged by noise, so the second axis is a second measurement of the same grid
and not a second dispatch table.

The noisy reference gradient is
`[-1.943050739368e-01, -5.290712169961e-01, 2.655714218122e-01]`, i.e.
`[1.388e-02, 3.779e-02, -1.897e-02]` from the unnoisy one. The three deterministic
noisy methods differ by at most `1.135e-11`, but that spread is `finite_difference`'s
displacement rather than a disagreement about the derivative: `parameter_shift` and
`autograd` agree to `1.110e-16`, and each differs from `finite_difference` by the
`1.135e-11`. Both numbers are recorded so that a reader can neither mistake the noisy
rows for a repeat of the unnoisy ones nor read the spread as an accuracy claim for the
shift rule. The gate recomputes the spread, because a number nobody re-derives is how a
disagreement goes unnoticed.

### `auto` is a policy, not a dispatch entry

`auto` has no per-mode behaviour of its own; it reads the loss at `parameters` and
reports what it resolved to. All three resolutions were reproduced:

| condition | resolves to | `exact` |
| --- | --- | --- |
| the loss carries a graph | `autograd` | `true` |
| no graph, `loss=` given | `parameter_shift` | `true` |
| no graph, no `loss=` | `finite_difference` | `false` |

Because it reports the method that ran, `auto`'s cells record
`reported_method = "autograd"` under the reference program.
`reported_method_is_the_method_that_ran` states that, so nobody reads the printed method
as an echo of the request.

### Cost is measured, and `spsa` is not the cheap option it looks like

Loss evaluations for the reference program's three parameters, counted rather than
derived:

| method | loss evaluations | scales with |
| --- | --- | --- |
| `autograd` | 1 | one forward and one backward pass, independent of the parameter count |
| `parameter_shift` | 6 | two per declared shift-rule term |
| `finite_difference` | 7 | two per parameter |
| `spsa(directions=1)` | 3 | two per direction, independent of the parameter count |
| `spsa(directions=4)` | 9 | two per direction |
| `spsa(directions=32)` | 65 | two per direction |

`spsa` is an estimator over Rademacher directions, so `[accuracy]` records a range over
64 seeds rather than a value, and the median is the mean of the two central order
statistics of an even-sized sample:

| directions | min | median | max | loss evaluations |
| --- | --- | --- | --- | --- |
| 1 | `4.927e-01` | `8.514e-01` | `8.514e-01` | 3 |
| 2 | `2.845e-01` | `5.669e-01` | `8.514e-01` | 5 |
| 4 | `3.448e-11` | `2.845e-01` | `8.514e-01` | 9 |
| 8 | `3.448e-11` | `2.458e-01` | `6.380e-01` | 17 |
| 32 | `2.242e-02` | `1.147e-01` | `3.018e-01` | 65 |

The reading rule recorded in the file is the `exact` flag, and this table is why: at
`directions=4` the best of 64 seeds is `3.448e-11`, which is as good as the deterministic
method, and the median at the same setting is `2.845e-01`. A single seed is not an
accuracy claim. Spending 65 evaluations on `directions=32` still leaves the median at
`1.147e-01`, against `3.749e-11` for `finite_difference` at 7 evaluations. Nothing
measured here makes `spsa` competitive with the deterministic method on this program, and
no cell claims it is.

### One reframing, recorded rather than corrected

`parameter_shift` wraps the program it cannot shift in `CapabilityError: the
parameter-shift rule does not apply to this program: <inner message>`. The wrapper
catches `ValueError`, and `ValidationError` is a `ValueError` while `CapabilityError` is
not. So the same inner failure is reported two ways depending on the method:

```text
parameter_shift  -> CapabilityError: the parameter-shift rule does not apply to this
                    program: mode must be one of: auto, density_matrix, mps, ...
autograd         -> ValidationError: mode must be one of: auto, density_matrix, mps, ...
```

Both halves are measured for two instances -- an unreachable mode name, and a noise model
with a mode other than `auto`/`density_matrix` -- and the gate asserts the asymmetry
still holds and still does **not** hold for `autograd`, `finite_difference`, or `spsa`.
The diagnosis is imprecise in the shifted case, because the rule does apply and the
loss's arguments are what is wrong, but it is recorded rather than corrected: changing
documented exception behaviour in a protected module needs its own authorization, and
this change is a measurement.

### What the file does not claim

- No accuracy claim for a method that is not in `serving_modes` with a serving cell.
- No claim that `spsa` reaches the exact gradient at any measured budget.
- No claim about `parameter_shift` correctness beyond the declared shift-rule terms;
  that boundary belongs to
  [`parameter-shift-coverage-contract.toml`](../../contracts/parameter-shift-coverage-contract.toml).
- No row for `adjoint`, and no row for a distributed mode, because neither is reachable
  from a user-supplied `fq.gradient` call.
- No noise column: noise is an axis, and its measured effect is recorded per cell.

### Using it

The file is a lookup, and the lookup is what a user needs before choosing a method:

```python
import torch
import flagquantum as fq

parameters = torch.tensor([0.3, 0.7, -0.4], dtype=torch.float64)

def build_circuit(values):
    return (fq.Circuit(2).ry(0, theta=values[0]).rx(0, theta=values[1])
            .ry(1, theta=values[2]).cx(1, 0))

# The mode is chosen where the program is executed, not where it is differentiated.
def loss(circuit):
    return fq.run(circuit, options=fq.ExecutionOptions(mode="mps"),
                  outputs=fq.expectation(fq.Z(0))).expectations[0]

result = fq.gradient(build_circuit, parameters, loss, method="parameter_shift")
print(result.method, result.exact, result.step)
```

`mps` with `parameter_shift` is a serving cell, so this runs. `stabilizer` with any
method is not, and the contract says so before the run rather than after it.

## Consequences

- A user can answer "which method can my loss use" from one file, and the answer is
  re-measured on every push instead of trusted.
- Adding a gradient method or an execution mode now fails the gate until the matrix is
  revisited: both vocabularies are read from the implementation, so a new method or mode
  leaves a pair that no cell describes.
- The three falsified predictions are in the file as refusals and as the
  `unreachable_modes` list, so the reasons travel with the matrix.
- The reframing is now visible. It was previously undiscoverable without reading
  `flagquantum/gradients.py`.

## Open questions

1. **Should the reframing be corrected?** Two options: give the shift wrapper a narrower
   `except` so it stops absorbing `ValidationError`, or leave it and rely on this record.
   Correcting it changes documented exception behaviour in a protected module and needs
   its own authorization.
2. **Should the five distributed mode names stay unreachable?** They are accepted by the
   runtime's mode tables and refused by `ExecutionOptions`. Either the option layer
   should accept them -- in which case this contract gains five columns, and gains them
   only when a distributed gradient is real -- or the runtime tables should stop
   advertising a name no caller can pass.
3. **Should `spsa`'s default `directions=1` be reconsidered?** The measured median at the
   default is `8.514e-01` deviation from the exact gradient, which is far from useful,
   while `directions=4` costs six more evaluations and moves the median to `2.845e-01`.
   The default is a deliberate user choice, but it is not a documented one.

## Owner and approvals

- **Implementation owner:** core (`flagquantum/gradients.py` as the behaviour measured,
  `flagquantum/runtime/options.py` as the mode vocabulary).
- **Integration surfaces touched:** `contracts/` (one new contract),
  `.github/workflows/ci.yml` (one step), `tools/` (one gate, one pre-push entry),
  `docs/api-changes/README.md` (one index line). These are protected paths, so the change
  is an integration change and carries its own contract, gate, tests, and record.
- **Authorization:** `docs/api-changes/FQ-GRADIENT-METHODS-MATRIX-20261027.md` (this
  document), listed in the contract's `authorization` field.
- **No API-owner approval required.** The change adds no stable export, and changes no
  signature, default, result field, or exception behaviour. It measures what is already
  there.
