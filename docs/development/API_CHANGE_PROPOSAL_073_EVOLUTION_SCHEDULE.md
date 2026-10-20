# API Change Proposal 073: Piecewise-constant Hamiltonian controls

## Status and review scope

**Approved for implementation by the repository owner on 2026-10-20.** The owner
explicitly approved proposal 073, requested merging #643, and authorized M1 time
control implementation in the working session. #643 merged as
`41cbf7433f37cd3edb697f6326f249d13ad62693` after all 36 checks and both full
workflows passed. The approval and replaceable static-composition test fake landed in #645 as
`df2f308de855e2101a13a53c1afde604cc7126f0`, after all 36 checks and both full
workflows succeeded. The implementation is now under review; approval alone
is not a capability claim. This revision updates
the contract against M1's static mixed-dimension and operator-coefficient work
(PRs [639](https://github.com/flagos-ai/FlagQuantum/pull/639),
[640](https://github.com/flagos-ai/FlagQuantum/pull/640), and
[641](https://github.com/flagos-ai/FlagQuantum/pull/641)). Approval of proposal
078 does not approve this temporal extension. This document changes no API,
serialized payload, capability status, or numerical implementation.

The decisions to review are: one binding per output interval; reuse of existing
full-space operator terms and parameter binding; all three current integrators;
a scheduled-only plan version 1.5 with unchanged static 1.3/1.4 compatibility;
live coefficient gradients and initial-density adjoints; and the acceptance
checks below. Implementation must pass the acceptance gates before being claimed
as available.

## User outcome and measured reuse

A user supplies fixed finite-level or boson operators and changes their scalar
coefficients between declared times. Three-level driving and truncated spin-mode
exchange use the same plan/run, density result, and PyTorch workflow. This is a
piecewise-constant mathematical model, not a continuously evaluated pulse or a
hardware control program.

The existing owners already supply the necessary building blocks:

| Existing boundary | Reused responsibility |
| --- | --- |
| `core/parameters.py` | `Parameter`, `ParameterExpression`, binding, scalar conversion, and parameter-name discovery. |
| `core/ir/state_space.py` | Ordered finite product dimensions; subsystem 0 most significant. CircuitIR 1.0 is unchanged. |
| `simulation/lindblad_request.py` | Static operator/coefficient terms, matrix and boson conversion, dimension/finite/Hermiticity checks, state/collapse/observable normalization. |
| `simulation/lindblad_generator.py` | Existing Liouvillian and SuperOperator arithmetic for each bound interval. |
| `simulation/lindblad_integrators.py` | `advance_interval(derivative, state, interval, ...)` and all three current methods. |
| `lindblad/_plan.py` | Canonical numerical requests, identity, decision verification, live bindings, and detached JSON replay. |
| `simulation/lindblad_adjoint.py` | Reverse propagation of an initial-density cotangent, including seeds at every output time. |

A current-API feasibility experiment at `ad62590f968014596830669f20d2d087fa46ae61`
composed static `fql.run` intervals; it did not implement a scheduled entry point.
Using Krylov exponential with the default local target `1e-12` on
Python 3.10 / Torch 2.13, CPU complex128, a three-level transition with
`omega=(1, 3, 7)`, `times=(0, 0.1, 0.3, 0.35)`, and `H_k=omega[k]*X12/2`
produced final level-2 probability `0.251214476054136`, differing from
`sin(sum(omega[k]*dt[k])/2)**2` by `1.6653345369377348e-16`. The largest
coefficient-gradient error against the analytic derivative was
`1.734723475976807e-17`. These are feasibility measurements, not scheduled-API
acceptance results or portable machine-precision guarantees.

Two ordered qutrit pi pulses, first on levels 0/1 and then 1/2, gave final
`P2=0.9999999999999991`; reversing the complete `(Hamiltonian, duration)` pairs
gave `P2=0`. With fixed level-2 decay rate 0.2, the ordered calculation differed
from an independently constructed dense-superoperator exponential reference by
`2.3314683517128287e-15` in maximum density entry. Production code must use the
existing numerical path, not copy this small dense reference. For reproduction,
start from `rho0=|0><0|`, use `H0=pi*X01/(2*0.1)` for 0.1 and
`H1=pi*X12/(2*0.2)` for 0.2, where each X has unit off-diagonal entries only on
its named levels. The fixed collapse matrix is `C=|1><2|`, rate `gamma=0.2`.
The independent reference uses row-major vectorization and
`A=C.conj().T @ C` and
`Lk=-i*(Hk kron I - I kron Hk.T) + gamma*(C kron C.conj()
- (A kron I + I kron A.T)/2)`.
Its final density is `reshape(exp(L1*0.2) @ exp(L0*0.1) @ flatten(rho0))`.
This formula uses no package generator, SuperOperator, or integrator. The scheduled
fixed template can contain both X01 and X12, with bindings selecting one term on
each interval; no time-dependent operator leaf is needed.

Refining each of those same three physical intervals into 1, 2, and 4 equal
pieces gives the following current-static-API absolute population errors. RK/CN
use their defaults; the exponential row uses local target `1e-14`:

| Method | 1 piece/segment | 2 pieces/segment | 4 pieces/segment |
| --- | --- | --- | --- |
| Runge-Kutta | `1.5589878108568378e-05` | `1.0815162247990173e-06` | `7.056172557362572e-08` |
| Crank-Nicolson | `0.002316864885318598` | `0.000585057288372659` | `0.00014663485557142586` |
| Krylov exponential | `5.551115123125783e-17` | `2.220446049250313e-16` | `1.366873281227754e-11` |

RK/CN show their expected refinement behavior without a new integrator interface.
The exponential refinement observation is also a counterexample to interpreting
`solve_tolerance` as a global population-error guarantee: the last error remained
`1.366873281227754e-11` even with local target `1e-14`. A trial global `1e-12`
assertion failed; it is not recorded as passed or repaired by changing an existing
solver/test. The scheduled implementation must separately budget and verify
global numerical error, including the projected matrix-exponential arithmetic;
local defect acceptance alone cannot close that gate.

## User code

**Executable through the approved implementation under review.**
Only scalar coefficients are symbolic. Numerical PauliSumTerm and canonical
BosonTerm fields are not widened to contain expressions.

```python
import torch
import flagquantum as fq
import flagquantum.lindblad as fql

drive = fq.Parameter("drive")
omega = torch.tensor([1., 3., 7.], dtype=torch.float64, requires_grad=True)
transition = torch.zeros((3, 3), dtype=torch.complex128)
transition[1, 2] = transition[2, 1] = 1
plan = fql.plan(
    [{"operator": transition, "coefficient": 0.5 * drive}],
    [0., 1., 0.], [0., 0.1, 0.3, 0.35], dimensions=(3,),
    parameter_values=[{"drive": value} for value in omega.unbind()],
    method="krylov-exponential",
    options=fq.ExecutionOptions(require_gradients=True),
)
result = fql.run(plan)
(gradient,) = torch.autograd.grad(result.populations[-1, 2], omega)
# Target P2: 0.251214476054137
# Target gradient: [0.0433711612797009, 0.0867423225594017, 0.0216855806398504]
```

The same spelling accepts existing `BosonOperator` leaves with explicit
truncations, for example `dimensions=(2, 3)`. Users may compute complex phase
bindings with PyTorch before passing them; no expression operations or pulse DSL
are added. Bind conjugate coefficient pairs explicitly and validate the total
Hamiltonian on every interval.

## Input contract

Append optional keyword-only `parameter_values=None` to these existing entries,
preserving every current parameter's order and default:

- `flagquantum.lindblad.plan`, `run`, and `adjoint_gradient`;
- `simulation.lindblad.plan_density_matrix_evolution` and
  `evolve_density_matrix`;
- `simulation.lindblad_adjoint.adjoint_gradient`.

The accepted scheduled value is a finite `Sequence` of mappings from `str` or
`Parameter` keys to finite numeric scalars or zero-dimensional PyTorch tensors.
It is not a generator, callable, nested control batch, or single mapping. Existing
static calls with `None` keep their present validation, fast paths, and behavior.

1. Scheduled `hamiltonian` is the existing nonempty full-space term sequence:
   `{"operator": matrix_or_boson, "coefficient": scalar_or_parameter_expression}`.
   Fixed numeric terms may accompany symbolic terms. At least one coefficient
   contains a named parameter; otherwise bindings would be unused and are refused.
   Symbolic operators, Pauli descriptors, direct PauliSum inputs, and qubit
   selectors in these terms are outside this first scheduled contract. Refuse
   them rather than silently materializing a Pauli model. Static numeric Pauli
   execution remains matrix-free and unchanged.
2. Bind coefficient leaves through existing Core binding functions. Discover names
   through the existing parameter traversal. Do not create a second expression
   evaluator, parameter registry, or scheduled algebra type. Unbound symbolic
   terms without `parameter_values` retain the current static refusal.
3. Each interval mapping binds exactly the names used in the coefficients.
   Missing and unused names are errors with interval/name context. A Parameter
   key and string key normalizing to the same name are refused as duplicates,
   even if their values agree. Core's existing bind-method behavior is unchanged.
4. Every bound total Hamiltonian must be finite, full-space, and Hermitian within
   the existing precision tolerance. Individual terms need not be Hermitian.
   Reuse existing matrix conversion and boson truncation/byte limits; do not
   default dimensions, expand local qubit selectors, or pad to powers of two.
5. The fixed initial state, collapse operators/rates, observable requests, device,
   precision, and batch meaning retain their existing contracts. All initial-state
   batch members share one control sequence. No parameter-batch axis is inferred.

## One time grid and one generator per interval

`times` remains the sole output and control-boundary grid. It is finite and
strictly increasing, may start away from zero, and may have nonuniform spacing.
For a scheduled request require at least two times. Exactly `len(times)-1`
bindings are required: entry `k` is held on `[times[k], times[k+1])`. The initial
state is recorded before any propagation; the final state is recorded after the
last interval. No first/last binding is discarded, repeated, or carried forward.
States remain continuous at a boundary; changing a coefficient is not an
instantaneous gate or measurement event.

For interval `k`, bind and validate `H_k`, then use the existing Liouvillian with
the same fixed collapse operators. Call the existing `advance_interval` with
that generator's derivative and `dt_k`. **Its derivative signature and integrator
implementations do not need a time argument.** Every substep sees the same
interval generator; switch only at the next declared boundary.

All existing methods are admitted:

| Method | Meaning on each declared interval |
| --- | --- |
| `runge-kutta` | Existing two fourth-order substeps; refine the grid to assess integration error. |
| `crank-nicolson` | Existing two second-order substeps with the existing matrix-free solve/residual policy. |
| `krylov-exponential` | Existing exponential action of the frozen interval generator; `order=None`, accuracy controlled by the existing defect target. |

The former draft's refusal of RK/CN was unjustified for this model: coefficients
are constant inside each integration interval and every discontinuity is already
on the grid. Retain the existing method defaults, order, substep count,
`solve_tolerance` validation, and non-convergence refusal. Do not silently select
another method. Keep generator residuals distinct from time-discretization error
and from a user's approximation of a continuous waveform. A local defect target
is not a guaranteed global state or population error bound.

There is one trajectory assembly and one result path. A small private extraction
of the existing validated interval loop is allowed if needed for both static and
scheduled callers; a second solver/loop copied alongside it is not. Calling the
public planner/run API once per interval is a reference technique, not the
production implementation.

## Plans, serialization, and gradients

Only scheduled requests use `flagquantum.lindblad_plan` **1.5**. Static qubit
plans remain 1.3 and static non-qubit plans remain 1.4, with byte-identical
payloads, identities, readers, and trajectories for unchanged requests. A
scheduled request is not folded into a static plan even if every binding agrees.

The 1.5 canonical request stores the **ordered bound numerical Hamiltonians**
in `request.hamiltonians`, one per interval, instead of `request.hamiltonian`.
Other request data keep their existing forms: normalized initial density, times,
fixed collapse operators/rates, observables, and density-retention flag. Qubit
`n_qubits` and non-qubit ordered `dimensions` follow the present 1.3/1.4
canonicalization; non-qubit `n_qubits` is null. No second state-space vocabulary
is introduced.

The decision adds `numerics.time_dependence="piecewise_constant"` and
`numerics.n_intervals=len(times)-1`, only for 1.5. If the existing EvolutionPlan
summary needs corresponding fields, append optional default-None fields and
omit them from static `to_dict()` output. EvolutionResult fields and axes do not
change. `trajectory_bytes` still estimates density history, not total memory;
storing interval matrices additionally costs `N*D*D*element_bytes`, shared across
initial-state batches. No new total-memory-budget support is claimed.

Use existing canonical JSON, hashing, and decision reconstruction. Verify matrix
count/order/shapes, dimensions, controls metadata, and the rebuilt decision even
when an attacker recomputes the outer hash. Reject unknown/mismatched 1.5
semantic fields and incompatible versions; do not reject valid static 1.3/1.4
payloads. Never accept scheduled-only fields under those static versions.

Bindings and expressions are compilation inputs, not a second serialized source
of truth. Equal ordered numerical generators and request settings yield equal
identities, even if parameter names or algebraically redundant bindings differ.
Changing an effective interval generator, its order, grid, or space changes the
identity. The old draft's assertion that every binding-value change necessarily
changes identity is therefore replaced by this execution-equivalence rule.

A live plan retains the normalized interval tensors and existing autograd links.
A restored 1.5 plan feeds verified numerical interval data into the same private
Simulation execution boundary; it must not synthesize dummy parameters, rebuild
symbolic source, or add a public replay-only keyword. There is no extra result,
plan wrapper, or runtime registry. Closed `fql.run(plan)` refuses all semantic
overrides, including `parameter_values`, using the existing TypeError pattern.

Autograd must accumulate coefficient gradients across every interval, including
shared tensor bindings and expressions, and preserve existing initial-state
batching/gradients. Rebind and rebuild plans after optimizer updates as in the
static workflow. JSON contains values, not graphs: restored plans requesting
gradients retain the existing execution-time `gradients_not_available` refusal.
No control-gradient availability is inferred merely from a tensor-shaped input.

Initial-density adjoints reuse the existing no-graph forward/backward path.
Reverse interval `k` with the adjoint of the same forward generator `H_k`, its
original positive duration, and all output-cost seeds. Reverse interval order,
not just time labels; do not reuse the last generator for the whole sweep.
The return is still a cotangent for the normalized initial density matrix,
not a coefficient gradient. Preserve the existing `adjoint_requires_no_graph`
refusal when trajectory-gradient options are requested on that route.

## Early refusals and unsupported scope

Validate every binding and bound generator before integration/trajectory history
allocation, using only existing bounded normalization work. Do not claim that
planning allocates nothing. New scheduled refusals use EvolutionValidationError:

| Condition | Code / owning field |
| --- | --- |
| Invalid sequence, insufficient times, wrong interval count, or no symbolic names | `invalid_parameter_values` / `parameter_values` |
| Missing, unused, duplicate-normalized, or invalid binding key | `invalid_parameter_binding` / `parameter_values[k]`, naming interval and key |
| Boolean, non-finite, non-scalar, or unbindable coefficient value | `invalid_coefficient` / interval binding or coefficient field |
| Unsupported Hamiltonian template/callable/Pauli scheduled input | `unsupported_time_dependence` / `hamiltonian` |
| Gradient-bearing scheduled time boundaries | `unsupported_time_gradient` / `times` |
| Bound Hamiltonian is non-Hermitian or has invalid shape | Existing numerical validation code with interval/term context |

Reuse existing device, precision, initial-state, observable, byte-ceiling,
solve-tolerance, and non-convergence errors. Static invalid requests keep their
existing behavior. No interpolation, callable controls, event/measurement
feedback, moving atoms, hardware pulse emission, time-dependent collapse rates,
time-boundary optimization, photonic backend, GPU, or distributed dynamics are
part of this contract. No Schedule class, new IR hierarchy, pass manager,
trainer, dependency, or Stable Core root export is authorized.

## Acceptance before claiming the feature implemented

Approval and implementation evidence are separate gates:

- [x] Owner approves the input, versioning, gradient, and method decisions above.
- [x] Contract-first change and conformance fake exercise the existing Simulation
      input/plan boundary, followed by the real implementation with unchanged
      consumer assertions. Do not create a second contract registry.
- [x] One public call reproduces the nonuniform qutrit reference and analytic
      per-interval gradients above at complex128; complex64 has a separately
      precision-justified bound. Validate boson exchange through the same path.
- [x] The noncommuting two-pulse witness distinguishes reversed intervals;
      compare a dissipative case against an independently vectorized
      `torch.matrix_exp` reference, not package Liouvillian or integrator code.
- [x] RK/CN converge under refinement that repeats bindings while preserving
      each physical control segment. Check exponential/implicit residual limits
      and a real non-convergence refusal. Quantify the exponential numerical floor
      against an independent reference; never relax existing accuracy tests.
- [x] Compare constant-binding schedules to existing static evolution, while
      checking their intentionally distinct serialized control model.
- [x] Compare live coefficient gradients to analytic and finite-difference
      references; reuse a tensor across intervals; perform an actual optimizer
      update. Verify exact live/JSON behavior and missing-graph refusal.
- [x] Compare initial-density adjoints with autograd for a noncommuting,
      dissipative schedule and costs reading several output times, including
      an initial-state batch. Confirm the sweep does not retain a control graph.
- [x] Verify all early refusals, per-interval Hermiticity, endpoint/binding counts,
      arbitrary time origin, no ignored entries, and no implicit batch/control
      axes. Each guard needs a counterexample that fails when it is removed.
- [x] Preserve the checked-in 1.3 fixture and measured 1.4 identities/trajectories.
      Round-trip 1.5 and reject forged counts, order, dimensions, metadata,
      unknown semantic fields, and scheduled data disguised as static payloads.
- [x] Check ordered-generator identity changes and equal-physical-request identity
      equivalence; closed plans refuse later binding overrides.
- [x] Prove static consumers still use the current integrators and numeric Pauli
      fast path; code review finds no duplicate integration or parameter engine.
- [x] Run supported-Python focused regressions, pr-default and pr-runtime tiers,
      typing/style and repository/API checks. Record revision, counts, numerical
      errors, and unsupported scope; update existing capability/parity sources
      and generated views only after implementation evidence exists.

## Implementation verification

The implementation in [PR 647](https://github.com/flagos-ai/FlagQuantum/pull/647)
was verified at source commit
`2ef01320419e922a46edda28ada5048ba7f5bb74`:

- 431 focused Lindblad/conformance/closed-subspace cases passed on each of
  Python 3.10 / Torch 2.13, Python 3.11 / Torch 2.14.1, and Python 3.12 / Torch 2.13.
- Full smoke/unit tier: 13,461 passed, 175 skipped on Python 3.10 / Torch 2.13.
- Full integration tier: 1,703 passed, 127 skipped on Python 3.12 / Torch 2.13.
- Mypy under repository defaults passed for 771 source files; Black, Ruff, API/IR and repository
  architecture, dependency, capability, documentation, parity, language and
  ownership checks passed. The PR example output matched its expected text;
  `python -m examples.train_piecewise_drive` completed 80 Adam updates and
  printed `Training reduced loss: True`.

The full tiers require local loopback communication. Restricted-sandbox attempts
hit socket/Gloo refusals and are not counted as passes. A supplemental Python
3.10 integration run also exposed an existing QEC mixed-check refusal-message
failure; it reproduces on unchanged baseline
`df2f308de855e2101a13a53c1afde604cc7126f0` with that interpreter. No QEC code or
assertion is changed here. The acceptance integration run uses Python 3.12,
matching the existing runtime lane. Remote CI remains a separate merge gate.

The first remote CI run rejected the call to PyTorch's untyped `Function.apply`
in `matrix_free_autograd.py`. The original local mypy invocation used repository
defaults, so reporting it as strict validation was incorrect. The follow-up gives
that existing autograd boundary its exact callable signature, following the
existing JAX bridge pattern, without changing numerical execution. The exact CI
command, `mypy --strict --python-version 3.12 --ignore-missing-imports flagquantum`,
then passed for 771 source files with mypy 2.4.0 and Torch 2.13; all 125 focused
closed-subspace, piecewise-control and contract cases passed. Remote dependency
floor and quality results remain separate acceptance evidence.

After synchronizing main, coverage shard 0 on `6291f61771` exposed an invalid
cross-invocation bitwise comparison in the closure test: separate complex64
GMRES least-squares calls returned an imaginary cancellation remainder of
`-1.4964e-17` and zero. The follow-up snapshots each kernel result immediately
before its gradient correction and compares the correction output against that
same invocation byte for byte, in both graph modes. The test also compares both forward modes and the independent dense value
numerically; independent dense derivative oracles and their original tolerance
budgets remain unchanged. No production kernel or
solver tolerance changes for this test repair.

## Closed-subspace sensitivity found during implementation

An inactive first drive, followed by an active one, exposed an existing static
kernel issue: the Arnoldi/GMRES forward space closed exactly, but perturbing
that drive leaves the closed space. Static composition reproduces it, so a
schedule-layer workaround would leave the shared numerical bug in place.
For `H_k=g_k*X/2`, `g=(0,1)`, `times=(0,.1,.3)`, and initial `|0>`, the
pre-fix first-control gradients were `0.004962655704877462` (CN) and `0`
(exponential). The corrected derivatives are `0.009925311409754924` and
`0.00993346653975306`, matching the discretized finite difference; the latter
also equals `.05*sin(.2)`. These new observations do not rewrite the earlier
recorded experiments or their status.

The owning matrix-free kernels now apply a first-order residual-sensitivity
correction at exact closure, preserving forward value bits. GMRES reuses GMRES
on the adjoint action. The exponential reuses its existing action on a block
operator with an `n+m` vector; it forms no dense vectorized generator. The
small private autograd glue has these two concrete consumers and owns no solver,
IR or parameter semantics. Independent complex64/complex128 dense derivative
oracles, zero-input and insufficient-backward-budget witnesses are in
`tests/unit/test_matrix_free_closed_subspace_gradients.py`. The static and
scheduled conformance consumers both include the inactive-drive regression.
Only first-order reverse-mode closure corrections are certified; higher-order
requests explicitly fail. Existing non-closed Krylov/autograd conditioning
limits remain disclosed, rather than being claimed repaired by this change.

## Runnable composition reference using the static API

This block demonstrates the independent static-composition reference retained
by the contract-fake tests. The production scheduled path uses one shared
trajectory loop rather than these repeated public calls.

```python
import torch
import flagquantum as fq
import flagquantum.lindblad as fql

drive = fq.Parameter("drive")
omega = torch.tensor([1., 3., 7.], dtype=torch.float64, requires_grad=True)
times = [0., 0.1, 0.3, 0.35]
transition = torch.zeros((3, 3), dtype=torch.complex128)
transition[1, 2] = transition[2, 1] = 1
state = torch.tensor([0., 1., 0.], dtype=torch.complex128)
for k in range(3):
    result = fql.run(
        [{"operator": transition,
          "coefficient": (0.5 * drive).bind({"drive": omega[k]})}],
        state, times[k:k+2], dimensions=(3,),
        method="krylov-exponential", return_density_matrices=True,
        options=fq.ExecutionOptions(require_gradients=True),
    )
    state = result.density_matrices[-1]
(gradient,) = torch.autograd.grad(result.populations[-1, 2], omega)
print(f"Final level-2 probability: {result.populations[-1, 2].item():.8f}")
print([round(value, 8) for value in gradient.tolist()])
# Final level-2 probability: 0.25121448
# [0.04337116, 0.08674232, 0.02168558]
```

## Historical measurements retained from the previous draft

The following numbers are retained as historical observations, not remeasured
scheduled-API evidence. The original document is available in
[the pre-revision source](https://github.com/flagos-ai/FlagQuantum/blob/b15fd13a5e9196d7d41f206a1f56e0b4c8ff4ccd/docs/development/API_CHANGE_PROPOSAL_073_EVOLUTION_SCHEDULE.md).
It used a qubit, `H_k=omega_k*X/2`, `omega=(1,3,7)`, uniform `dt=0.1`, and
Krylov solve tolerance `1e-12`:

| Recorded quantity | Historical value |
| --- | --- |
| Three chained intervals, P0 | `0.726798060712789` |
| Closed-form P0 | `0.7267980607127886` |
| Absolute difference | `3.331e-16` |
| Single-interval constant error | `5.551e-16` |

The prior draft also recorded the following wrong-rule comparisons. Their labels
and numbers are preserved without treating them as independently rerun oracles;
the new noncommuting/analytic tests must establish their own counterexamples.

| Historical wrong-rule label | Recorded predicted P0 | Recorded margin |
| --- | --- | --- |
| First coefficient held across the grid | `0.9776682446` | `2.509e-01` |
| Coefficients summed but dt applied once | `0.9667637743` | `2.400e-01` |
| Coefficient held at t=0 | `1.0000000000` | `2.732e-01` |

The earlier schema-1.4 assignment, rejection of old 1.3 plans, symbolic
PauliSumTerm widening, integrator time-argument change, and RK/CN prohibition
are superseded proposal decisions, not retained compatibility requirements.
