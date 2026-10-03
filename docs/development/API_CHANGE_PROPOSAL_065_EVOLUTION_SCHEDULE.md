# API Change Proposal 065: A coefficient schedule for time-dependent evolution

## Status

**Proposed. No code is written by this proposal and no behaviour changes until
the repository owner approves it and the acceptance items below are measured.**

This proposal defines the contract for evolving a state under a Hamiltonian
whose coefficients change with time. It is the contract W4-05 names and the
prerequisite W5-13 needs before `evolve` can be offered as a counterpart to
`cudaq.evolve`, so leaving it unwritten keeps two plan rows blocked rather than
one.

Four premises below were measured on this checkout before the proposal was
written, because engineering decision principle 6 requires inspecting what
already exists and decision principle 10 forbids claiming a gap that has not
been observed. They are marked where they appear. Everything in "Required
evidence" is a requirement on the implementation, not a result: nothing here has
been built.

## Problem

### 1. The generator is built once, from a Hamiltonian frozen at construction

`evolve_density_matrix` and `plan_density_matrix_evolution` in
`flagquantum/simulation/lindblad.py` both build the generator before the
interval loop and never rebuild it:

```python
generator = Liouvillian(hamiltonian_representation, collapse, hilbert_dimension=dim)
for start, stop in zip(time_grid[:-1], time_grid[1:], strict=True):
    rho, residual = advance_interval(generator.derivative, rho, stop - start, ...)
```

Measured: `Liouvillian.derivative` has the signature `(self, state)` and reads
`self._hamiltonian`, which is the value passed to the constructor. There is no
time argument anywhere on the integration path — `advance_interval` takes
`derivative: Callable[[torch.Tensor], torch.Tensor]`, `_runge_kutta_4_step`
calls `derivative(state)` four times without a time, and
`_krylov_exponential_interval` builds its `apply` closure around the same
single-argument callable.

So the time dependence is not merely unwritten; the interface between the
planner and the integrators has nowhere to put it. A coefficient cannot vary
across the trajectory however it is spelled at the call site.

### 2. A callable coefficient is structurally impossible today

The natural spelling of a time-dependent Hamiltonian — a coefficient that is a
function of time — cannot even be constructed. Measured:

```
>>> PauliSumTerm(coefficient=lambda t: 1.0, mask=1)
TypeError: complex() first argument must be a string or a number, not 'function'
```

`PauliSumTerm.coefficient` is typed `complex` and its `__post_init__` coerces
with `complex(self.coefficient)`, and `PauliSum.__post_init__` then materializes
every coefficient into a tensor. There is no room for a symbolic or callable
coefficient in the matrix-free Hamiltonian either.

### 3. What CUDA-Q actually does, and what must not be inherited

`cudaq.evolve` takes `schedule` as its third positional argument and refuses
anything that is not a `cudaq.Schedule` instance:

```python
if not isinstance(schedule, Schedule):
    raise ValueError(f"Invalid argument `schedule` for target {target_name}.")
```

Reading the shipped implementation rather than the docstring, the semantics are
exact and narrower than the name suggests:

- the schedule is an iterable of parameter mappings, and **the time grid is the
  schedule's own** — `tlist = [schedule.current_step for _ in schedule]`. There
  is no separate `times` argument;
- step 0 is skipped and its mapping is never evaluated, so a schedule of `n`
  mappings produces `n - 1` evolution steps;
- each step applies `expm(-i * H(parameters_k) * dt_k)` with `dt_k = tlist[k] -
  tlist[k-1]`, so the Hamiltonian is **frozen for the whole step** and the step
  propagator is the exact exponential of that frozen operator. Steps may be
  non-uniform;
- `schedule.reset()` is called once per member of a batch, so the schedule is a
  stateful iterator rather than a value.

Two of those properties must not be inherited. A silently ignored first entry is
a rule that cannot be checked from the output, and a stateful iterator that must
be reset is a second source of truth for a value the plan also has to record.
The frozen-generator-per-step rule and the per-step `dt` are the parts worth
having, and both are properties of the **arithmetic**, not of the name
`Schedule`.

### 4. The composition rule is already reachable through the existing API, and
it is exact

The arithmetic this proposal asks for can be assembled today by chaining one
`evolve_density_matrix` call per interval, which is what makes its reference
value measurable now rather than at implementation time. Measured on one qubit
with `H_k = 0.5 * omega_k * X`, `omega = (1.0, 3.0, 7.0)`, `dt = 0.1`, starting
from `|0><0|` and using `method="krylov-exponential"` with
`solve_tolerance=1e-12`:

| Quantity | Value |
| --- | --- |
| measured `P0` after the three chained intervals | `0.726798060712789` |
| closed form `cos²(sum(omega) * dt / 2)`, `sum(omega) * dt = 1.1` | `0.7267980607127886` |
| difference | `3.331e-16` |

Rotations about one axis compose by adding their angles, so the reference needs
no matrix product and no numerical reference implementation — it is one `cos`
call. Against the same measured value, three plausible wrong rules miss by
`2.4e-1` to `2.7e-1`:

| Wrong rule | Predicted `P0` | Margin |
| --- | --- | --- |
| the first coefficient held across the whole grid | `0.9776682446` | `2.509e-01` |
| the coefficients summed but `dt` applied once | `0.9667637743` | `2.400e-01` |
| the coefficient held at its `t = 0` value | `1.0000000000` | `2.732e-01` |

The single-interval constant case is accurate to `5.551e-16` on the same
fixture, so `3.331e-16` is the composition step's own error and not a floor
inherited from the integrator.

## Decision

### 1. The schedule is a list of bindings, not a new type

FlagQuantum already has the vocabulary this needs and it is in the Stable Core:
`Parameter` and `ParameterExpression` (`flagquantum/core/parameters.py`) express
a coefficient symbolically, `ParameterExpression.bind(values)` resolves it, and
`Parameter.bind(values)` refuses a missing name with
`KeyError("Missing value for parameter 'theta'.")` rather than defaulting.

So a time-dependent Hamiltonian is a `PauliSum` whose `PauliSumTerm.coefficient`
is a `ParameterExpression` over named `Parameter`s, and the schedule is the
sequence of mappings those names are bound to, one per interval. No
`Schedule` class is introduced. A new class would be the fifth name for a
concept the repository already has four of (`Parameter`, `ParameterExpression`,
`bind_parameters`, and the plan's recorded request), and decision principle 11
forbids that.

`PauliSumTerm.coefficient` therefore widens from `complex` to
`complex | ParameterExpression`. The second half of that sentence needs care,
because `PauliSum.__init__` is where the values are turned into numbers:
measured, it builds `self._coefficients = tuple(torch.tensor(
term.action_coefficient(), ...) for term in self._terms)` at construction time
and caches the result. A symbolic coefficient therefore cannot simply be stored
and resolved later; `PauliSum` has to keep the binding question open until it is
asked a numeric question.

The smallest change that admits this is for `PauliSum` to accept either a bound
coefficient or a `ParameterExpression` and to refuse any numeric operation while
one is unbound, naming the parameter as `Parameter.bind` already does. The
per-interval `PauliSum` the integrator needs is then a `bind` call away, which is
what decision 3 relies on. The alternative — a separate scheduled-Hamiltonian
type carrying a coefficient table — is a second representation of the same
object and is rejected for that reason.

### 2. The time grid stays `times`; the schedule has one entry per interval

The plan's request already records `times` and the plan identity is
`sha256(canonical_json(request + decision))`, so the grid is already a recorded
part of the decision. Introducing a schedule that carries its own grid would
create a second source of truth for the same fact, and the two could disagree.

The rule is therefore:

- `times` keeps its existing meaning: the output grid, strictly increasing, one
  more entry than there are intervals;
- `parameter_values` is an optional sequence of mappings with exactly
  `len(times) - 1` entries, the value bound on interval `k = [times[k],
  times[k+1])`;
- a sequence of the wrong length is refused, naming both lengths. This is the
  deliberate deviation from CUDA-Q's `n` mappings for `n - 1` steps: an entry
  that is never evaluated cannot be distinguished from an entry that was
  supposed to be evaluated and was dropped.

A caller who wants non-uniform steps gets them for free, because `times` is the
grid and `dt_k` is read off it.

### 3. One frozen generator per interval, advanced by the matrix-free
exponential

`advance_interval` gains the interval's start time and a derivative that accepts
it, so the generator can be rebuilt per interval from the bound coefficients.
The integrator that matches the CUDA-Q semantic is `krylov-exponential`: one
matrix-free action of the frozen generator's exponential per interval, which is
`expm(-i * H * dt)` applied without forming the `16**n` superoperator. The
existing matrix-free Pauli-sum path is reused unchanged; a bound
`ParameterExpression` produces a `PauliSum` with numeric coefficients, which is
exactly what the current code already consumes.

### 4. `runge-kutta` and `crank-nicolson` are refused for a schedule

Both schemes' accuracy claims are statements about a **constant** generator over
the interval they integrate: `runge-kutta`'s fourth order and
`crank-nicolson`'s second order are both derived by Taylor expansion in `dt` with
`H` fixed. Applying either to a generator that changes inside the interval
silently changes the order the plan reports, and the plan reports the order.
That is a wrong claim, not a wrong number.

So a request that supplies `parameter_values` together with either explicit
scheme is refused at planning time with the method named and
`krylov-exponential` offered as the one scheme whose accuracy is set by
`solve_tolerance` rather than by a step-size order. A caller who wants an
explicit scheme can quantise the schedule finely enough that the coefficient is
constant to within the tolerance they care about, but they have to say so by
adding grid points rather than by relying on an order claim that does not hold.

### 5. A missing binding is refused, never defaulted

Every parameter that appears in any coefficient must be bound on every interval.
A mapping that omits one is refused naming the parameter and the interval index.
Defaulting it to zero, to its previous interval's value, or to its value at
`t = 0` are all the same defect: the returned trajectory would be that of a
different Hamiltonian, and no property of the output would show it.

### 6. The schedule is recorded in the plan decision, and gradients follow the
input form

The plan decision gains `numerics.time_dependent` and `numerics.n_intervals`,
and the bound values are recorded in the request, so two plans that differ only
in their schedule have different identities. The plan schema
`flagquantum.lindblad_plan` moves from `1.3` to `1.4` for the added decision
fields, in the same way `1.3` recorded the declared batch axis.

Gradients follow the same rule the existing path already measured: coefficients
supplied as `torch.Tensor` scalars in process stay connected to the graph, and a
plan restored from JSON holds numbers rather than tensors, so
`require_gradients=True` on a restored plan is refused with the existing
`gradients_not_available` reason rather than returning a trajectory whose
derivative is silently zero.

### 7. Fail closed at planning time

Every refusal below is raised by `plan_density_matrix_evolution` before any
state is allocated, so an unrepresentable request costs nothing and reports
which field owns the objection:

| Condition | Refused with |
| --- | --- |
| `parameter_values` length differs from `len(times) - 1` | both lengths, `parameter_values` |
| a coefficient's parameter is unbound on some interval | the parameter name and the interval index |
| `parameter_values` with `runge-kutta` or `crank-nicolson` | the method name, `method` |
| `parameter_values` supplied but no coefficient is symbolic | `parameter_values`, because a schedule that binds nothing is a mistake |
| an unbindable coefficient type (a callable) | the type, `hamiltonian` |

## Public API

One optional keyword argument on two existing entry points, one widened field
type, and two new decision fields. No Stable Core root export is added,
removed, or renamed: `plan_lindblad_evolution` and `evolve_density_matrix` are
not among the 34 names in `docs/public_api_v1.json`, so `docs/public_api_v1.json`
and `contracts/public-api-v1-candidate.json` are unchanged by this proposal.

```python
from flagquantum import Parameter
from flagquantum.lindblad import plan
from flagquantum.simulation.matrix_free_hamiltonian import PauliSum, PauliSumTerm

drive = Parameter("drive")
hamiltonian = PauliSum(
    [PauliSumTerm(coefficient=0.5 * drive, mask=1)],
    n_wires=1,
)

planned = plan(
    hamiltonian,
    [[1.0, 0.0], [0.0, 0.0]],
    [0.0, 0.1, 0.2, 0.3],
    parameter_values=[{"drive": 1.0}, {"drive": 3.0}, {"drive": 7.0}],
    method="krylov-exponential",
)
planned.identity  # distinct from the same request with a constant coefficient
```

The names this proposal touches are `Parameter` and `ParameterExpression`
(unchanged, reused), `PauliSumTerm.coefficient` (widened), the
`parameter_values` keyword (new), and the two decision fields (new). The new
public name count is zero.

## Compatibility

- A request without `parameter_values` is byte-identical in behaviour: the same
  generator, the same integrators, the same trajectory. The constant path is not
  re-routed through the scheduled one.
- A `PauliSum` built with numeric coefficients is unchanged; the widened field
  type admits the values it already admitted.
- `times` keeps its meaning and its validation. No existing plan changes
  meaning, and the identity of an existing request is unchanged because no new
  field is serialized for it.
- A plan payload written at schema `1.3` is refused by the `1.4` build with the
  existing version-mismatch message rather than reconstructed. That is the same
  policy `1.2` → `1.3` used, and it is the honest one: the stored decision is
  rebuilt from its request and compared, so a payload that predates a decision
  field cannot be shown to mean what the new build would compute.
- The `parameter_values` keyword is added to a namespace that is not Stable
  Core, so no deprecation window is owed. The widened field type is a
  widening, so no caller is broken.

## Required evidence before this proposal can be accepted

Each item names the measurement, not the intention. A green suite does not show
that a guard is exercised, so each refusal is additionally proven by deleting
it and observing the test fail.

- [ ] The three-interval composition fixture in "The composition rule is already
      reachable" is reproduced **through the new `parameter_values` path in one
      call**, and `P0` matches `cos²(sum(omega) * dt / 2)` within `1e-12`.
      `1e-12` is the tolerance because the measured composition error is
      `3.331e-16` — four orders of magnitude of headroom — while the nearest
      wrong rule misses by `2.4e-01`, eleven orders of magnitude above it.
- [ ] The same fixture under each wrong rule is shown to fail the tolerance, so
      the tolerance is evidence rather than decoration.
- [ ] A schedule whose coefficients are constant in time reproduces the
      constant-Hamiltonian trajectory from the existing path at
      `dtype=torch.complex128` within `1e-12`, measured, with the two values
      reported.
- [ ] A time-dependent Hamiltonian **with** collapse operators is compared
      against a dense-superoperator reference built independently with
      `torch.matrix_exp` over the same piecewise-constant generator, at a
      tolerance derived from the measured difference between the two, not
      chosen in advance. The reference must be computed by a route that does not
      call the integrators under test.
- [ ] `krylov-exponential`'s reported residual under a schedule is bounded by
      `solve_tolerance`, and a tolerance that cannot be reached fails with
      `integration_not_converged` rather than returning the trajectory.
- [ ] Every refusal in the table under "Fail closed at planning time" is raised
      by a plan call, names the field it blames, and is proven by a mutation
      that deletes it and fails a test.
- [ ] Two plans differing only in `parameter_values` have different `identity`
      values, and one plan round-trips through `to_json`/`from_json` to the same
      identity.
- [ ] `require_gradients=True` with in-process tensor coefficients returns a
      trajectory that carries a graph, checked by a finite-difference comparison
      against a perturbed coefficient; and the same on a restored plan is
      refused with `gradients_not_available`.
- [ ] A `PauliSum` with an unbound `ParameterExpression` coefficient is refused
      by name before any state is allocated, and its message is the one
      `Parameter.bind` produces so the two cannot drift.
- [ ] The plan schema version is `1.4`, a `1.3` payload is refused with the
      version message, and `docs/public_api_v1.json` is unchanged.
- [ ] `python tools/ci_tier.py pr-runtime` and `python tools/ci_tier.py
      pr-default` pass, with the observed counts recorded here.
- [ ] The capability matrix records the new support boundary: the
      time-dependence part of the dynamics row moves only as far as the measured
      evidence allows, and `docs/reference/KNOWN_LIMITATIONS.md` states that
      `runge-kutta` and `crank-nicolson` do not accept a schedule.
- [ ] Repository owner approves. Not yet requested; see `## Status`.

## Non-goals

- A public `Schedule` type, a stateful iterator, or a `reset()` protocol. The
  bindings are a value, and the plan records them.
- Interpolation between schedule entries, or a callable schedule. CUDA-Q's
  schedule is piecewise-constant, and a caller who wants finer resolution adds
  grid points, which is visible in the plan identity; interpolation inside an
  interval would be a second, unrecorded rule about how the coefficient moves.
- An `integrator` extension point. The plan row `W4-05` is a contract for
  representing time dependence, and a pluggable integrator protocol is a
  separate boundary with its own replacement test.
- Qudits, `dimensions`, or the `dynamics`-target path. Those are the
  `cuDensityMat` counterpart (`W4-04`, `W4-07`), and the dimension mapping is a
  second axis of the same contract rather than part of this one.
- Analytic (imaginary-time) evolution, adiabatic schedules, pulse-level control,
  or operator-sum schedules.
- `evolve` itself. `W5-13` builds the counterpart to `cudaq.evolve` and reuses
  whatever this proposal settles; it is not this proposal's deliverable, and no
  root export is added here.
- Any distributed or multi-device claim. The scheduled path is a single-device
  CPU path, like the constant one it extends, and `scalability_claim_allowed`
  stays false.
