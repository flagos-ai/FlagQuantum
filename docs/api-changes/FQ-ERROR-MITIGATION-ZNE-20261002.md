# Zero-noise extrapolation as a native capability

## Decision and authorization

Status: **proposed, not approved.** This document records what the capability is,
where it lives, what it may and may not claim, and the evidence for it. It
contains no claim of approval.

This is a port of the in-flight commit `8b56047c feat(algorithms): add zero-noise
extrapolation with declared assumptions` — W6-09's first vertical slice on
`refactor/flagquantum-vnext-architecture`, which was never merged. Only the
zero-noise commit moves. The branch it sits on is a larger in-flight change:
`feat/pennylane-alignment` at `f47aba11` differs from `origin/main` in 188 files,
while `8b56047c` itself changes 18. The port lands on `main` and the content is
unchanged except for one correction recorded under **Compatibility**.

The arrival window is real but narrow, and it is not a claim that another
framework lacks the capability. PennyLane 0.45.1 ships zero-noise extrapolation in
`pennylane.noise`: `mitigate_with_zne`, `fold_global`, `poly_extrapolate`,
`richardson_extrapolate`, `exponential_extrapolate`, `NoiseModel` and
`add_noise`. PennyLane `main` at `0.46.0-dev112` **removes the whole module**,
recorded in `doc/releases/changelog-dev.md` as "The `pennylane.noise` module has
been removed, including `NoiseModel`, `add_noise`, `insert`, noise mitigation
transforms (`mitigate_with_zne`, `fold_global`, `poly_extrapolate`,
`richardson_extrapolate`, `exponential_extrapolate`), and `from_qiskit_noise`"
([PR #10214](https://github.com/PennyLaneAI/pennylane/pull/10214)). The
capability exists there at a released version and not on its next one; the
comparison below is against 0.45.1, and it is the reason this slice is worth
carrying rather than the reason it is novel.

Scope of the affected surface, all of it additive:

| Path | Owner | Change |
| --- | --- | --- |
| `flagquantum/algorithms/error_mitigation.py` | `algorithms` | New module, 780 lines |
| `flagquantum/algorithms/__init__.py` | `algorithms` | Re-exports the new names |
| `flagquantum/algorithms/README.md` | `algorithms` | Domain README entry |
| `capability-maturity.toml` | `integration` (protected) | New `[capabilities.algorithms_zero_noise_extrapolation]` |
| `contracts/cudaq-parity-matrix.toml` | `integration` (protected) | `unsupported` → `partial` for error mitigation |
| `docs/generated/CAPABILITIES.md`, `docs/reference/KNOWN_LIMITATIONS.md` | generated | Regenerated from the capability record |
| `docs/guides/ALGORITHMS.md` | shared | The user-facing guide |
| `docs/roadmap/NOISY_SIMULATION_ROADMAP.md`, `docs/roadmap/CUDAQ_PARITY_STRATEGY.md`, `docs/reference/CUDAQ_PARITY_MATRIX.md` | shared | Boundary statements |
| `examples/algorithms/error_mitigation.py`, `examples/**/README.md` | shared | The ten-minute golden path |
| `tests/unit/test_error_mitigation.py`, `tests/test_error_mitigation.py`, `tests/test_algorithm_examples.py` | shared | Focused and scenario tests |
| `flagquantum/kernels/README.md` | `simulation` | Records why the unit is not a kernel |

No public stable name is added or removed. `fq.__all__` is unchanged,
`docs/public_api_v1.json` is unchanged, and `python -m tools.public_api_snapshot`
passes, because the new names live in `flagquantum.algorithms` rather than at the
`fq.*` root.

## Problem and affected user journey

### The pieces exist and the workflow does not

A FlagQuantum user can already build a noisy model, simulate it exactly, and read
an observable off it:

```python
from flagquantum.circuit import Circuit
from flagquantum.noise import NoiseModel, depolarizing_channel
from flagquantum.algorithms import Hamiltonian, HamiltonianTerm

circuit = Circuit(2).h(0).cx(0, 1)
observable = Hamiltonian([HamiltonianTerm(1.0, "zz", (0, 1))])
model = NoiseModel().add("cx", depolarizing_channel(0.05))
density = circuit.noisy_density_matrix(model)
value = float(observable.expectation(density))
```

There is no entry point that scales that model, measures the curve, and continues
it to zero noise. A user who wants the mitigated number has to write the loop by
hand, and — this is the part that matters — has no place to record the assumption
the resulting number rests on. The loop is six lines and the assumption is the
whole difficulty: `estimate` is unbiased exactly when the measured curve is a
polynomial of the fitted degree in the scale factor, and nothing in a hand-rolled
loop says whether that held.

The capability record agrees that the gap is real and wide:
`contracts/cudaq-parity-matrix.toml` recorded error mitigation as `unsupported`
with the reason "Absent. VQE-class optimization exists in the algorithms domain,
but no error-mitigation technique does."

### What the user journey is after this change

```python
import torch

import flagquantum as fq  # noqa: F401 - the namespace the user imports
from flagquantum.algorithms import Hamiltonian, HamiltonianTerm, run_zne
from flagquantum.circuit import Circuit
from flagquantum.noise import NoiseModel, depolarizing_channel

circuit = Circuit(2).h(0).cx(0, 1)
observable = Hamiltonian([HamiltonianTerm(1.0, "zz", (0, 1))])
model = NoiseModel().add("cx", depolarizing_channel(0.05, dtype=torch.complex128))

result = run_zne(
    circuit,
    observable,
    noise_model=model,
    scale_factors=(1.0, 3.0, 5.0, 7.0),
    order=2,
    dtype=torch.complex128,
)

result.estimate                # 1.000000003030
result.fit.max_residual        # 4.1723e-09
result.fit.variance_amplification  # 2.940625
result.unmitigated             # 0.871111109257
result.assumptions             # the assumption chain the estimate rests on
result.limitations             # what the estimate does not cover
```

The estimate is the least interesting field. `max_residual` is the diagnostic
that exposes a violated assumption, `variance_amplification` says what the fit
would cost a shot-based estimate of the same points, and `assumptions` and
`limitations` travel with the result rather than sitting in a footnote.

### What is different from the PennyLane 0.45.1 shape

PennyLane expresses zero-noise extrapolation as a tape transform whose folding and
extrapolation are both caller-supplied:

```python
from pennylane.noise import mitigate_with_zne, fold_global, poly_extrapolate

mitigate_with_zne(
    tape,
    scale_factors=[1.0, 3.0, 5.0, 7.0],
    folding=fold_global,
    extrapolate=poly_extrapolate,
    extrapolate_kwargs={"order": 2},
)
```

measured on 0.45.1 as `mitigate_with_zne(tape, scale_factors, folding,
extrapolate, *, folding_kwargs=None, extrapolate_kwargs=None,
reps_per_factor=1) -> tuple[Sequence[QuantumScript], Callable[[Sequence[Result]],
Result]]`. The transform returns tapes and a postprocessing function; the
assumption is documented separately in the transform's docstring and is not a
field of what comes back. The noise is inserted by a second transform,
`add_noise(tape, noise_model, level="user")`, against a `NoiseModel` whose
entries are predicates over operations.

Three differences are deliberate and are the substance of this proposal:

1. **The noise is scaled by a declared rule, not by circuit folding.**
   `fold_global` grows noise by repeating the circuit, which scales *every* error
   source at once and is defined only for a circuit. FlagQuantum multiplies the
   one error-probability parameter each channel declares, so the scaled object is
   a `NoiseModel` at a different strength and the claim being made is inspectable;
   a channel where that multiplication has no meaning is refused by name instead
   of being folded and mislabelled. Circuit folding arrives separately as N6-11
   and is not part of this unit.
2. **The assumption is a field of the result, not a docstring.**
   `ZneResult.assumptions` and `ZneResult.limitations` are non-empty by
   construction, and the two-claim distinction — the offered scaling versus a
   caller's callable, which transfers the claim to the caller — is recorded per
   result.
3. **Readout error is refused rather than silently omitted.**
   This unit reads `Tr(O rho)`, which is before measurement. A model that declares
   a readout rule is refused, because extrapolating a curve that omits readout
   confusion would return a state-preparation estimate under the name of a
   measured one.

The first two are the same shape as PennyLane's and are not claimed as better.
The third is a refusal PennyLane's transform path does not make, because there the
noise model and the readout layer are separate concerns.

## Decision Candidates

**Candidate 1 — A unit in `flagquantum.algorithms` that returns its own
diagnostics. (chosen)**
`run_zne(circuit, hamiltonian, *, noise_model, scale_factors, order, ...)` returns
a `ZneResult` carrying the estimate, the `ExtrapolationFit`, the measured curve
with each point's model identity, the unmitigated measurement, and the assumption
and limitation chains. Scaling is a separate public function,
`scale_noise_model(model, factor)`, so a caller can inspect or replace exactly the
step whose validity is in question.

**Candidate 2 — Put it in `flagquantum.noise` beside the model it scales.**
The scaling rule is a property of the noise model, so this is where the code is
closest to its cause. It also puts an observable-estimation workflow in the noise
domain, which owns the representation of noise rather than the estimation of a
quantity under it, and it would make the noise domain depend on the algorithms
domain's `Hamiltonian`.

**Candidate 3 — Express it as an `fq.Module` and let `fq.train` drive it.**
The framework's training path is the flagship entry point, so a mitigation unit
that plugs into it would be the most native-looking. It is also wrong for this
slice: extrapolation is not a differentiable objective, the fit is a least-squares
solve on a handful of points, and modelling it as a module would advertise a
gradient path that does not exist.

**Candidate 4 — Implement circuit folding first (N6-11) and extrapolate after.**
Folding is the noise-amplification mechanism a reader expects to see first, and
PennyLane's shape leads with it. It also makes the first slice depend on a
mechanism whose scale factor is defined per gate sequence, which is the larger
piece of work, and leaves nothing measurable end to end until it lands. The plan
orders these the other way for that reason.

## Prohibited Practices

1. **Returning an estimate without the residual that qualifies it.** A number from
   a fit whose residual is above the rounding floor is a number the method has no
   claim on. A result that carries `estimate` and not `max_residual` would be read
   as a correction.
2. **Reporting an arithmetic zero for a square fit's residual.** A fit with no
   degree of freedom left interpolates, so its residual is zero by construction
   and is not evidence. The record reports it as absent rather than as `0.0`, and
   a later change must not "simplify" that to zero.
3. **Scaling a channel by multiplying a parameter that is not an error
   probability.** An over-rotation angle, two independent reset probabilities, or
   a relaxation time constant scaled by a factor changes the *shape* of the noise,
   not its amount. Those channels are refused by name; a caller who wants one
   supplies a `scaling` callable and owns the claim, and the result records which
   of the two claims it rests on.
4. **Measuring a model that declares a readout rule.** This path cannot represent
   classical readout confusion, so measuring anyway would silently drop it and
   report a state-preparation estimate as a measured one.
5. **Reporting a shot-based uncertainty for an exact-expectation estimate.**
   `variance_amplification` is a property of the scale grid and the degree alone;
   calling it a variance would invent a measurement that never happened.
6. **Adding a second scaling implementation.** `scale_noise_model` and the
   `scaling` parameter are the one implementation and its replacement point. A
   second private scaler would be the parallel source of truth that engineering
   decision principle 6 forbids.

## Compatibility

- **Additive, and outside the Stable Core.** `flagquantum.algorithms` re-exports
  eight new names — `run_zne`, `scale_noise_model`, `extrapolate_polynomial`,
  `extrapolate_richardson`, `ZneMeasurement`, `ZneResult`, `ExtrapolationFit`
  and the `error_mitigation` module itself. `ExtrapolationMethod`,
  `EXTRAPOLATION_METHODS`, `richardson_weights`, `ZNE_ASSUMPTIONS` and
  `ZNE_LIMITATIONS` stay in `flagquantum.algorithms.error_mitigation` and are
  reachable there without being re-exported. None is exported from
  `flagquantum/__init__.py`; `fq.__all__` stays at 35 names and
  `docs/public_api_v1.json` is untouched.
- **`capability-maturity.toml` gains a row** at level `development_evidence`, and
  `contracts/cudaq-parity-matrix.toml` moves error mitigation from `unsupported`
  to `partial` with a `maturity_ref` pointing at that row. Both are protected
  integration surfaces: this document is the contract change that precedes them,
  and the implementation does not proceed without it.
- **The parity totals move by one.** 38 `partial` and 48 `unsupported` where the
  generated scoreboard said 37 and 49. The document is regenerated from the
  contract, not edited.
- **The support boundary narrows nothing.** No existing entry point changes
  behavior, no exception type changes, and no capability row loses a level.
- **One inherited measured claim is corrected.** The ported guide, test comment,
  capability record and example all quoted a single-precision estimate distance
  next to a double-precision unmitigated distance, and the example computed its
  noiseless reference outside the path the estimate took, at a precision the fits
  did not run at: `Circuit.density_matrix()` takes no dtype and
  returns the runtime's default precision. Those figures come from different runs
  and different references. `1.288888907432559e-01` is the double-precision
  distance from the analytic value `1.0`, `1.3e-7` is the single-precision
  distance from the same analytic value, and the exactly simulated noiseless read
  the test asserts against is `0.9999999403953552` — itself a single-precision
  quantity, `6.0e-8` away from `1.0`. Measured single precision against that
  reference: estimate `6.78002827214641e-08` away at a residual of
  `6.258487716959138e-08`, unmitigated `1.2888896465301514e-01` away, an
  improvement of `1.901009250691689e+06`. Measured `complex128` against its own
  noiseless read `0.9999999999999998`: estimate `3.0299036613001817e-09` away at a
  residual of `4.172325152040912e-09`, unmitigated `1.288888907432557e-01` away,
  an improvement of `4.2538940e+07`. The texts and the example's reference now
  name what they are measured from. The example measures its reference with
  `run_zne` on an empty `NoiseModel`, so the reference and the estimate come from
  the same entry point at the same dtype rather than from a private runtime import
  that no public surface offers a dtype for. No number was re-derived and the
  extrapolation itself is unchanged.

## Acceptance Tests

1. **A polynomial family is recovered and its residual says so.** On
   `H(0); CX(0,1)` with `ZZ` and depolarizing noise at `p = 0.05` on the `cx`, the
   four scale factors `1, 3, 5, 7` give the exact curve `0.871111109257`,
   `0.639999987284`, `0.444444444444`, `0.284444452922`; degree one leaves
   `1.7777786784701877e-02` at estimate `0.951111100846`, degree two leaves
   `4.172325152040912e-09` at estimate `1.000000003030`, and the noiseless value
   is `0.9999999403953552`.
2. **A violated assumption stays visible.** Under a coherent over-rotation of
   `0.15`, the declared scaling refuses the channel by name, and a caller-supplied
   scaling gives estimates `1.271993498172` at degree one and `1.105716958201` at
   degree two, with residuals `8.932955226566808e-02` and
   `2.886535591251327e-02` — orders above the `4.2e-9` floor, so the residual
   separates a wrong family from a right one.
3. **A readout rule is refused before anything is simulated.** The refusal
   precedes scaling and simulation; a test proves it by recording that no scaling
   call happened.
4. **A square fit reports its residual as absent, not as zero.**
5. **A scaled model is a model, not a circuit.** `scale_noise_model` returns a
   `NoiseModel` whose identity differs from the original and whose parameters are
   the declared ones multiplied, and the channels it refuses are enumerated by
   name in the module's limitations.
6. **Precision moves the floor, not the conclusion.** In the default single
   precision the estimate is `6.78002827214641e-08` from the exactly simulated
   noiseless read and the improvement is `1.901009250691689e+06`; in
   `complex128` the estimate is `3.0299034e-09` from the analytic value `1.0`.
7. **The example is executable and asserts its own claims.**
   `python examples/algorithms/error_mitigation.py` runs the three sections
   (polynomial family, wrong assumption, readout boundary) and exits non-zero if a
   refusal it demonstrates stops refusing.

## Open Questions

1. **Where does folding live?** N6-11 adds gate and global folding. It scales a
   circuit rather than a model, so it may not belong in this module — but a
   `scale_factors` grid is shared, and a second scaling concept beside
   `scale_noise_model` is the duplication prohibited above.
2. **Should the estimate carry a confidence interval once shots exist?**
   `variance_amplification` is the factor a shot-based estimate would be
   amplified by. Converting it into an interval needs a shot count and a
   variance model, neither of which this slice has, and inventing them would make
   the exact-expectation result look sampled.
3. **Does `order` need to be chosen rather than supplied?** The guide says the
   degree is chosen by residual and not by ambition, and the caller currently
   chooses. A residual-driven selection is a decision rule the unit does not own,
   because the same residual profile can mean a wrong family rather than a wrong
   degree.
4. **What is the model-device agreement check?** N6-13 proposes validating the
   noise model against a `DeviceNoiseProfile` before extrapolating. This unit
   scales whatever model it is given and says nothing about whether that model is
   the device's; the two are separate claims and the second is not made here.

## Owner and approvals

- Owning domain: `algorithms` for `error_mitigation.py` and the example, with
  `integration` for the capability record and the parity contract, and
  `simulation` for the note in `flagquantum/kernels/README.md` explaining why this
  workload needs no fused kernel.
- Required approvals before implementation: **algorithms domain owner** (the unit
  and its public names), **integration owner** (the capability row, the parity
  transition, and the generated documents), and **API owner** only if any of the
  new names is later promoted into `fq.__all__`, which this document does not
  propose.
- Verification: `python tools/check_team_scope.py --team integration --files
  <changed paths>` passes for the whole change set; `--team algorithms` reports
  the three protected and one simulation-owned path, which is the split recorded
  above. `python tools/ci_tier.py pr-default` and `python tools/ci_tier.py
  pr-runtime` are the tiers for a new local algorithm on the density-matrix path.
