# A native SPSA optimizer for objectives with no gradient

## Decision and authorization

Status: **proposed, not approved.** This document records what the capability is,
where it lives, what it may and may not claim, and the evidence for it. It
contains no claim of approval.

This adds one class, `flagquantum.algorithms.SPSAOptimizer`, and nothing else.
It lives in `flagquantum.algorithms` and is not promoted to the `fq.*` root, so
no Stable Core name, signature, default, result field, or serialized schema
changes: `fq.__all__` stays at 34 names and `docs/public_api_v1.json` is
untouched. The exact-name row of the parity contract is what moves, and that is
the reason this document exists rather than the code being the reason.

The capability is not novel and the proposal does not claim it is. PennyLane
0.45.1 ships `qml.SPSAOptimizer(maxiter=None, alpha=0.602, gamma=0.101, c=0.2,
A=None, a=None)` with `step`, `step_and_cost`, `compute_grad`, and
`apply_grad`, measured from `pennylane/optimize/spsa.py`. This unit's
`estimate_gradient` is the same computation as that `compute_grad`, named for
what it returns rather than for what it is not. CUDA-Q ships `cudaq.optimizers` with an SPSA
entry, which is the baseline this repository is measured against. The gap is
recorded and open: `contracts/cudaq-parity-matrix.toml` has
`optimizer_library` at `status = "partial"` with the reason "The gradient-free
half is the gap, because COBYLA, NelderMead, and SPSA have no implementation and
the training entry point refuses an optimizer that is not a PyTorch one."

Scope of the affected surface, all of it additive:

| Path | Owner | Change |
| --- | --- | --- |
| `flagquantum/algorithms/spsa.py` | `algorithms` | New module |
| `flagquantum/algorithms/__init__.py` | `algorithms` | Re-exports `SPSAOptimizer` |
| `flagquantum/algorithms/README.md` | `algorithms` | Domain README entry |
| `docs/guides/ALGORITHMS.md` | shared | The per-unit index row and a section |
| `docs/reference/API.md` | shared | Records the `algorithms` surface |
| `examples/algorithms/spsa_optimizer.py` | shared | The ten-minute golden path |
| `tests/unit/test_algorithms_spsa.py`, `tests/test_algorithm_examples.py` | shared | Focused and scenario tests |
| `tests/api_contract/test_public_docstring_examples.py` | `compiler` (shared) | Adds the new entry to the doctest gate |
| `docs/api-changes/README.md` | shared | The proposal index entry |
| `capability-maturity.toml` | `integration` (protected) | New `[capabilities.algorithms_spsa_optimizer]` |
| `contracts/cudaq-parity-matrix.toml` | `integration` (protected) | The `optimizer_library` row narrows and gains a `maturity_ref` |
| `docs/generated/CAPABILITIES.md`, `docs/reference/KNOWN_LIMITATIONS.md`, `docs/reference/CUDAQ_PARITY_MATRIX.md` | generated | Regenerated from the capability record and the parity contract |

## Problem and affected user journey

### The framework can train, and only with an exact gradient

FlagQuantum's training path is PyTorch-first and that is a real advantage: the
statevector, MPS, and tensor-network runtimes differentiate natively, and
`fq.train` accepts a `torch.optim.Optimizer`. The parity contract records the
consequence in the same sentence as the advantage: "the training entry point
refuses an optimizer that is not a PyTorch one."

That refusal is correct and should stay. What is missing is the other half of the
optimizer library. A caller whose objective is a 4096-shot circuit measurement
on 40 parameters has no exact gradient to hand to Adam: parameter shift costs
`2 * 40` circuit evaluations per step and returns a noisy answer anyway. The
honest statement of the situation is not that SPSA beats gradient descent — it
does not, when a gradient exists — but that for a stochastic objective with many
parameters it costs two evaluations per step instead of `2 * n`.

The user journey today ends here:

```python
>>> import torch, flagquantum as fq
>>> fq.SPSAOptimizer
AttributeError: module 'flagquantum' has no attribute 'SPSAOptimizer'
```

### What the user journey is after this change

```python
import torch

import flagquantum as fq
from flagquantum import algorithms as fqa

def sampled_energy(values):
    circuit = fq.Circuit(2, dtype=torch.complex128)
    circuit = circuit.ry(0, values[0]).ry(1, values[1]).cx(0, 1)
    outputs = fq.expectation(fq.Z(0) + fq.Z(1))
    return -fq.run(circuit, outputs=outputs).expectation().sum()

optimizer = fqa.SPSAOptimizer(
    maxiter=120,
    perturbation=0.25,
    generator=torch.Generator().manual_seed(13),
)
values = torch.full((2,), 0.4, dtype=torch.float64)
for _ in range(120):
    values = optimizer.step(sampled_energy, values)

float(sampled_energy(values))   # -1.9996860765168998
optimizer.evaluations           # 240 -- two per step, whatever the parameter count
```

The objective is a plain callable, not an `fq.Module` and not a
`torch.optim.Optimizer`. That is deliberate: SPSA is for the case where the
objective is an external system — a sampled circuit, a queued hardware job, a
classical simulator behind a process boundary — and where attaching it to an
autograd graph is either impossible or a lie.

### Where this differs from the PennyLane shape

Both frameworks expose the same recursion and the same default exponents, and
that is not a defect: `alpha = 0.602` and `gamma = 0.101` are Spall's values and
the reason both implementations agree is that both follow the same paper. Three
differences are deliberate.

1. **The gain sequences are named for what they are.** PennyLane takes
   `alpha`, `gamma`, `c`, `A`, `a`. Four of those five are single letters whose
   meaning is only in a docstring. FlagQuantum takes
   `parameter_gain_exponent`, `perturbation_exponent`, `perturbation`,
   `stability`, and `parameter_gain`. Non-negotiable rule 9 requires a stable
   name to state its domain meaning, and the correspondence to the published
   symbols is recorded in the docstring so the paper can still be read against
   the code.

2. **The perturbation comes from a `torch.Generator` the caller owns.**
   PennyLane draws with `np.random.choice`, which is process-global state: two
   optimizers in one process interleave their draws and a test that seeds
   `numpy.random` is coupled to every other test in the session. FlagQuantum
   takes an explicit generator, so the trajectory replays from the call site and
   on the parameter's own device. Measured: seeding both with 5, 11, and 13
   produces the same sign stream from `np.random.choice` and `torch.randint`, so
   a caller migrating from PennyLane keeps its trajectory.

3. **The estimator is described as an estimator.** The module docstring states
   that the estimate is biased for every finite `c_k`, that it is a gradient
   approximation rather than a gradient, and that an objective with an exact
   gradient is served better and more cheaply by `torch.autograd` or by
   parameter shift. `optimizer.evaluations` is reported so a caller can compare
   that cost against the alternative rather than take it on faith.

## Decision Candidates

**Candidate 1 — A standalone optimizer class in `flagquantum.algorithms` that
takes a plain callable. (chosen)**
`SPSAOptimizer.step(objective, parameters)` and
`SPSAOptimizer.step_and_cost(objective, parameters)` follow the recursion, and
the cost is two evaluations per step by construction rather than by measurement.

**Candidate 2 — A `torch.optim.Optimizer` subclass, so `fq.train` accepts it
unchanged.**
The most native-looking surface, and wrong for this unit. A PyTorch optimizer
consumes `p.grad`, which assumes something already computed a gradient; SPSA's
whole point is that it computes the descent direction itself, from objective
values. Implementing it as one would mean either running the objective inside
`step()` behind a closure the caller cannot see, or advertising a `grad`
contract that is never populated. `FQ-API-OPTIMIZER-FACTORY-20260910.md` already
records the `OptimizerFactory` protocol as the extension point for gradient-based
training; a gradient estimate that is not a gradient does not belong in it.

**Candidate 3 — A pure function, `spsa_step(objective, parameters, state)`.**
Everything is explicit and nothing is hidden, which suits a framework that
prefers deliberate code. It also makes the caller thread the iteration counter,
the generator, and the evaluation count through every call, and it puts the
gain-sequence arithmetic in the user's loop — the part of SPSA that is easiest to
get wrong and the part the capability exists to get right.

**Candidate 4 — Implement COBYLA and NelderMead first, since the parity row
names three optimizers.**
The row would move further on paper. Neither is a better fit for the case that
motivates the row: NelderMead's cost grows with the parameter count in exactly
the way SPSA's does not, and COBYLA is a constrained solver whose constraint
handling this repository has no user for. SPSA is the one of the three whose
evaluation cost is the reason to have it. The other two remain open gaps and
this document does not close them.

## Prohibited Practices

1. **Calling the estimate a gradient.** The difference quotient is centered on
   a finite perturbation, so its expectation is the gradient only in the limit.
   The module says so once, in the docstring, and no name or error message
   claims otherwise.
2. **Presenting SPSA as the preferred optimizer.** For an objective with an
   exact gradient, autograd is cheaper and exact. Docs that lead with SPSA, or
   that rank it beside Adam without the cost comparison, would be a claim this
   evidence does not support.
3. **Adding an `fq.*` root export.** The root surface is Stable Core and
   protected. A promotion is a separate change with its own proposal; if it ever
   happens it is `N5-5`'s, whose subject is the training entry point.
4. **Adding a second perturbation source.** `_perturbation` is the only place
   signs are drawn. A "use numpy when the caller has no generator" fallback would
   be a second source of truth and would make the drawn stream depend on an
   unrelated global.
5. **Averaging the estimates, or any other variance-reduction knob, without a
   measured reason.** The recursion as published updates from one estimate. Any
   additional smoothing is a second algorithm with its own convergence
   conditions, and it is not added here.
6. **Tolerating an objective that writes into the tensor it is handed.** SPSA
   evaluates the objective at two perturbed points and reads the reported cost
   at the unperturbed one; an in-place objective makes those three evaluations
   describe different parameters. The refusal is a `ValidationError`, not a
   silent clone, because a silently cloned parameter would leave the caller's
   own tensor mutated and the reported cost wrong.

## Compatibility

- **Additive, and outside the Stable Core.** `flagquantum.algorithms` re-exports
  one new name, `SPSAOptimizer`, and the `spsa` module itself. Nothing is
  removed, renamed, or reordered, no default changes, and no documented exception
  behavior changes.
- **`fq.__all__` is unchanged at 34 names** and `docs/public_api_v1.json` and
  `contracts/public-api-v1-candidate.json` are not touched. `python -m
  tools.public_api_snapshot` passes without a rewrite.
- **The doctest gate gains an entry.** `SPSAOptimizer`'s `Examples:` block is a
  `>>>` example in a new module, and
  `tests/api_contract/test_public_docstring_examples.py::test_every_module_with_examples_is_covered`
  fails on any example no listed entry runs. Adding it to `ENTRIES` is what makes
  the example executable rather than decorative.
- **`capability-maturity.toml` gains a row** at level `development_evidence`, and
  `contracts/cudaq-parity-matrix.toml`'s `optimizer_library` row narrows its
  reason to the two optimizers that remain absent and gains a `maturity_ref`
  pointing at that row. Both are protected integration surfaces: this document is
  the contract change that precedes them.
- **The support boundary narrows nothing.** No existing entry point changes
  behavior, and no capability row loses a level.

## Acceptance Tests

All figures below were measured on this branch at `torch 2.14.0 / Python
3.12.14`, dtype `torch.float64` for parameters and `torch.complex128` for
circuits, from `flagquantum.algorithms.SPSAOptimizer` with a seeded
`torch.Generator`. The unit's own suite is 39 collected cases in
`tests/unit/test_algorithms_spsa.py`.

1. **The evaluation cost does not grow with the parameter count.** Ten `step`
   calls on a quadratic objective; the counts of objective calls are
   `20, 20, 20, 20, 20` for `1, 2, 8, 32, 128` parameters, and
   `optimizer.evaluations` agrees with the counted calls at every size.
2. **The estimate is an estimate, not a gradient.** Against the analytic
   gradient `[0.7480963877584119, 0.3586780454497614]` of the two-qubit Pauli
   energy at `[0.4, 0.4]` with `perturbation = 0.05`, one draw has relative error
   `0.9999`, the mean of 64 draws `0.1259`, and the mean of 512 draws `0.0284`.
   The mean converges to the gradient as `1/sqrt(n)` and a single draw is not it,
   which is why the module describes its output as an approximation of a gradient
   step. The same section measures the exact part: for a one-parameter quadratic
   the estimate is exact to `1e-9`.
3. **The gain sequences follow their exponents.** With `stability = 4`,
   `parameter_gain = 0.9`, `perturbation = 0.3`, `parameter_gain_exponent = 0.6`
   and `perturbation_exponent = 0.1`, iteration `k` reports
   `c_k = 0.3 / k**0.1` and `a_k = 0.9 / (4 + k)**0.6`, checked at every step of
   a seven-iteration run.
4. **A default `a_1` of 0.05 holds for any `maxiter`.** Measured
   `optimizer.step_size == 0.05` for `maxiter` in `10, 200, 5000`.
5. **The circuit workflow converges with no gradient.** On
   `RY(0) RY(1) CX(0,1)` against `-(Z0 + Z1)` from `[0.4, 0.4]`, 120 steps give
   `-1.9991819605521022`, `-1.9990938026576754`, and `-1.9996860765168998` for
   seeds 5, 11, and 13, at 240 evaluations each. PennyLane 0.45.1 on the same
   circuit, the same seeds, and `c = 0.25` reaches `-1.9991819605521028`,
   `-1.999093802657676`, and `-1.9996860765168996` over the same 120 steps and
   240 objective calls — the same three trajectories to about `1e-15`, because
   the seeded `torch.Generator` stream and the seeded `numpy` stream that
   PennyLane draws from are the same stream. That agreement is a measured
   observation and not an invariant this unit relies on: it pins no stream in a
   test, and the reason to prefer this optimizer is the caller-owned generator,
   not a different draw. See the guide's section for the
   `requires_grad` caveat that the PennyLane comparison has to satisfy.
6. **A shot-noisy objective converges to the exact minimum.** On a single-qubit
   `RY` against 4096 shots per evaluation, 200 steps from `0.35` reach
   `distance_to_minimum = 1.24e-05` (seed 17) and `1.55e-05` (seed 23) at 400
   evaluations.
7. **`step_and_cost` reports the pre-update cost.** On `(p**2).sum()` at
   `[1.0, 2.0]` it reports `5.0`, updates to `[1.1, 1.9]`, and spends three
   evaluations.
8. **A seeded generator replays and an unseeded one does not.** Two runs at
   seed 7 produce equal parameter vectors; seed 8 differs.
9. **Every refusal happens before the objective runs.** A refused call leaves
   both the caller's counter and `optimizer.evaluations` at zero, for the
   non-callable objective, the non-`Tensor` parameters, the integer-dtype
   parameters, the empty parameters, the non-finite parameters, the generator
   that cannot draw on the parameter device, and the objective that writes into
   the tensor it is handed.
10. **The example is executable and asserts its own claims.**
    `python examples/algorithms/spsa_optimizer.py` runs the cost comparison, the
    convergence, the shot-noisy run, and the refusals, and exits non-zero if a
    refusal it demonstrates stops refusing. Its printed lines are pinned by
    `tests/test_algorithm_examples.py::test_spsa_example_measures_its_cost_and_converges`,
    so the script fails a test rather than going stale.

## Open Questions

1. **Should the parity row name SPSA as closed and COBYLA and NelderMead as
   open?** The row's `partial` status is unchanged by this change and its reason
   now separates the closed half from the open one. A reader who wants one row
   per optimizer is asking for a matrix restructure, which is an `integration`
   decision and not part of this slice.
2. **Does `fq.train` ever accept this optimizer?** Only if the training path
   grows a notion of an optimizer that consumes an objective rather than a
   gradient. That is `N5-5`'s subject and it would be a `runtime` contract; this
   unit deliberately does not reach into it.
3. **Is a per-parameter perturbation scale needed?** SPSA perturbs every
   coordinate by the same `c_k`, which is wrong when the parameters carry
   different units or magnitudes. Spall's blockwise variant addresses it. No user
   has asked for it, the two-evaluation cost is lost if the blocks are separate
   optimizations, and it is not added speculatively.
4. **Should the module expose the estimate's variance across draws?** It can be
   measured — 64 draws gave `0.1259` relative error and 512 gave `0.0284` — but
   an interval needs a distributional assumption about the objective, which this
   unit cannot state for an arbitrary callable.

## Owner and approvals

- Owning domain: `algorithms` for `spsa.py`, its re-export, its README entry, its
  focused tests, and the example. `compiler` owns the doctest gate's `ENTRIES`
  tuple, which this change extends by one entry. `integration` owns
  `capability-maturity.toml`, `contracts/cudaq-parity-matrix.toml`, and the
  regenerated documents.
- Required approvals before implementation: **algorithms domain owner** (the
  unit, its parameter names, and its docstring claims), **integration owner**
  (the capability row, the parity row's narrowed reason, and the generated
  documents), and **compiler domain owner** for the one-line gate change.
  **API owner** approval is required only if `SPSAOptimizer` is later promoted
  into `fq.__all__`, which this document does not propose.
- Verification: `python tools/check_team_scope.py --team algorithms --files
  <changed paths>` passes for the algorithms-owned paths and names the
  protected integration paths and the `compiler`-owned gate as the split
  recorded above. `python tools/ci_tier.py pr-default` and
  `python tools/ci_tier.py pr-runtime` are the tiers for a new local algorithm
  on the statevector path.
