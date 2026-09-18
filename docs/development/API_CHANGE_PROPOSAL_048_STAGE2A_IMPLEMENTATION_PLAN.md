# QEC Detector Error Model (Stage 2a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `flagquantum/qec/dem.py` — an exact detector error model built from a `MemoryCircuit` and a phenomenological noise record, with parity matrices, exact marginal rates, seeded sampling, and a lossless stim text interchange.

**Architecture:** One new flat module beside the Stage 1 records layer. The model's error mechanisms are learned by *forcing* one physical error at a time and reading its detector and observable signature off the Stage 1 layouts through the unchanged private hybrid compiler — construction is exact and never samples, because the reference gate set is Clifford and every configured channel is a Pauli channel. A second, independent path (a per-shot injected sampler and an exhaustive pairwise linearity sweep) checks the model against the circuit simulator.

**Tech Stack:** Python 3.12 in-container, frozen dataclasses, `torch` tensors for parity matrices and samples, pytest (markers `unit` and `integration`), the private `flagquantum.compiler._hybrid` capture/lowering path, `flagquantum.runtime.dynamic.hybrid_session` execution.

**Spec:** `docs/development/API_CHANGE_PROPOSAL_048_QEC_DETECTOR_ERROR_MODEL.md` — read the **Problem**, **Layer placement and additive discipline**, **Stage 2**, **Noise scope**, **Public API**, **Interpretation**, and **Stage 1 landed state and known debt** sections before starting. Stage 1's own plan (`API_CHANGE_PROPOSAL_048_STAGE1_IMPLEMENTATION_PLAN.md`) is the precedent for structure and rigor.

---

## Global Constraints

- Work in the worktree `/Users/baai/Documents/liuwei/FlagQuantum-qec`. Never `cd` into the primary clone `/Users/baai/Documents/liuwei/FlagQuantum-upstream`; another session owns that working tree, and the `.git` directory is shared by every worktree.
- **Every command runs on a remote Docker host, never locally.** SSH to `jp-a800-172`; the image is `tovx/flagquantum:0.2.0-dev` (Python 3.12.13, pytest 9.1.1, black 26.5.1, ruff 0.15.21, mypy 2.2.0, torch 2.13.0, networkx 3.6.1, pytest-cov, pytest-xdist). Both `jp-a800-171` and `jp-a800-172` are shared multi-tenant A800 machines: only touch your own container, name it with the `fq-qec-claude-` prefix, and never `docker system prune` or remove another user's image.
- Sync the source tree from the worktree, then run inside the container:

  ```bash
  rsync -az --delete --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.pytest_cache' --exclude='.mypy_cache' --exclude='.ruff_cache' \
    --exclude='.superpowers' --exclude='.coverage' --exclude='coverage.xml' \
    --exclude='*.egg-info' --exclude='.venv' \
    ./ jp-a800-172:/root/liuwei/fq-qec-run/

  ssh jp-a800-172 'docker run --rm --name fq-qec-claude-<task> \
    -v /root/liuwei/fq-qec-run:/w -w /w -e PYTHONPATH=/w --entrypoint bash \
    tovx/flagquantum:0.2.0-dev -lc \
    "python -m pip install -e . --no-deps -q && <command>"'
  ```

  The image's own editable install points at a stale `/opt/FlagQuantum` snapshot, so the `pip install -e .` step is mandatory on every run; `-e PYTHONPATH=/w` makes submodule imports resolve to the mounted tree. `pytest-randomly` is not installed, so `-p no:randomly` is unnecessary. The image lacks `pennylane`, `hypothesis`, `datasets`, and `transformers`, so tests needing them skip.
- Do not modify `flagquantum/qec/types.py`, `decoders.py`, `repetition.py`, `pauli.py`, `codes.py`, or `circuit.py`. The seven names pinned by `contracts/hybrid-compilation-private-v0-candidate.json` keep their signatures.
- `flagquantum/qec/noise.py` **is** modified, but only additively: `RepetitionNoiseProfile` and `run_repetition_memory_noise_sweep` keep their current fields, signatures, and behavior.
- Do not modify `flagquantum/noise/`, `flagquantum/compiler/`, `flagquantum/runtime/`, `architecture.toml`, or `.github/`.
- Do not touch `capability-maturity.toml`, the generated docs under `docs/generated/`, or any capability row. Staging a capability promotion is Stage 5 work.
- Do not update `flagquantum/qec/README.md`. Its module list is stale from Stage 1 and `docs/development/API_CHANGE_PROPOSAL_048_QEC_DETECTOR_ERROR_MODEL.md` assigns the QEC documentation sweep to Stage 5.
- All source, docstring, and comment text is English (`tools/check_repository_language.py` gates this).
- Gates that must pass on every task that changes `flagquantum/`: `black --check`, `ruff check` (the repository's selected rule set requires `zip(..., strict=...)` in every `zip` call), `mypy --strict --python-version 3.12 --ignore-missing-imports flagquantum`, and `python tools/check_architecture.py`.
- Coverage floors in `contracts/coverage-policy.toml` must hold: `[global] min = 76` and `[packages.qec] min = 82`. New code must not lower either.
- Module line ceiling is 1250 for anything other than `flagquantum/__init__.py`. If `dem.py` crosses 1000 lines, stop and split the signature engine into `flagquantum/qec/_signatures.py` rather than pushing on.
- Every task ends with a commit. Conventional-commit prefixes used by this repository: `feat:`, `test:`, `docs:`.

---

## Design Decisions

These were settled with the user before this plan was written. They are decisions, not options — do not re-litigate them mid-implementation.

**1. Noise convention: textbook, not the frozen profile's.** Data flips are injected **once per data qubit at the start of each syndrome round**, before that round's CNOTs. This is the standard phenomenological convention and the one stim's `before_round_data_depolarization` uses, so the measured crossing can be compared against a published number, which is what the parity goal in the proposal asks for. It is deliberately *not* the frozen `RepetitionNoiseProfile` convention, which attaches a bit flip **after** each matching CNOT and therefore gives the middle data wire two opportunities per round.

**2. Scope splits into 2a and 2b, and this plan is 2a only.** 2a delivers `dem.py`, `PhenomenologicalNoise`, and the cross-check evidence. **2b** (a separate plan and PR) delivers circuit-level noise (`CircuitLevelNoise` plus a Kraus-to-Pauli decomposition) and is **construct-only** — it builds a model and proves its signatures, but does not run the "DEM-predicted rate vs circuit-sampled rate" comparison. That decision follows from a measured constraint, not a preference: `flagquantum/runtime/dynamic/_noise.py` fails closed with `"dynamic execution currently supports one-wire bit-flip channels only"` and `"must use independent readout rules"`, so a depolarizing channel cannot be sampled through the trajectory executor at all, and teaching it to would mean editing Runtime and Simulation, which the proposal's layer-placement discipline forbids.

**3. Error probabilities come from the noise record; signatures come from forced execution.** Construction is exact and does not sample. Propagating one Pauli mechanism through a Clifford circuit yields a deterministic detector and observable signature, so a single forced-error execution determines that mechanism's signature exactly. The builder never estimates a signature from shots.

**4. Every forced execution carries a determinism assertion.** Each mechanism is executed with `shots=2` and the two trajectories must agree bit for bit. That is simultaneously the evidence that the signature is exact rather than sampled, and a fail-closed guard against a non-Pauli or feedback-bearing program slipping through.

**5. Mechanisms that share a signature are merged; mechanisms with an empty signature are discarded and counted.** Merged probability is `p1 * (1 - p2) + p2 * (1 - p1)`. A mechanism that flips no detector and no observable has no effect on the model and must not appear as a `DemError`; the count of discarded mechanisms is retained so "no mechanism is missing or duplicated" stays assertable.

**6. Parity matrices and samples are `torch.int8`.** `detector_error_matrix()` has shape `(num_detectors, num_errors)` and `observables_flips_matrix()` has shape `(num_observables, num_errors)`, matching stim's orientation. `dem_sampling()` returns a `DemSample` record whose tensors have shape `(shots, num_detectors)` and `(shots, num_observables)`.

**7. Marginal rates are exact, not sampled.** `detector_rates()` and `observable_rates()` evaluate the closed form `(1 - prod(1 - 2 * p_e)) / 2` over the mechanisms touching each detector or observable. That formula is exact for any independent Bernoulli probabilities in `[0, 1]`, so no restriction on the merged probabilities is needed.

**8. stim text carries `detector` and `logical_observable` declaration lines.** Without them a model whose last detectors are touched by no error would lose its shape on a round trip. `to_stim_text()` therefore emits one declaration per detector and per observable, and `from_stim_text()` reads the shape back from those declarations.

**9. The DEM keys on check tuple position, never on `CodeCheck.index`.** The measurement record's classical-bit stride is `round_index * len(checks) + position_in_checks`, because that is what the lowerer emits. This is the first known-debt item Stage 1 recorded, and a test with a code whose declared indices disagree with their positions is required to prove it.

**10. The cross-check runs in the ordinary `integration` lane.** An earlier decision was to mark it `integration` and `slow` and keep it off pull requests. That is not achievable with markers alone: `slow` is not a selector any wired lane uses, `tests/unit/test_test_reachability_policy.py` does not list it in `LANE_SELECTORS`, and the `nightly` and `release` tiers are not invoked by any workflow. A `slow`-only test would be unreachable and would fail that policy check, while an `integration`-marked test runs in the `pr-runtime` and `coverage` jobs regardless of the `slow` marker. The cross-check is therefore made **cheap enough to run on every pull request**, by pairing a modest shot budget with an exhaustive deterministic linearity sweep that carries the stronger structural evidence.

**11. The cross-check is honest about what it proves.** The forced-error engine and the injected sampler share the injection helper, so the sampler does not independently re-derive the signatures. What it genuinely tests is the model's central assumption: that mechanisms compose by XOR in the real simulator, and that merging identical signatures yields the right marginal rates. The module docstring and `IMPLEMENTATION.md` must say so, and the one genuinely independent check available (parsing the emitted stim text with the real `stim` package) is a **developer-time verification, not a committed test**, because stim is not a dependency of this repository.

---

## Measured Budget

Measured on `jp-a800-172` in `tovx/flagquantum:0.2.0-dev` before this plan was written, on the code-driven memory circuit with `rounds = distance`. These are observations, not estimates — do not re-derive them, but do re-measure if a change makes the numbers move.

Forced-error execution (inject one X, `capture_source`, `lower_dynamic_program`, `execute_hybrid_dynamic_session(shots=1, strategy="trajectory")`):

| distance | mechanisms | distinct signatures | empty signatures | per mechanism | all mechanisms |
| --- | --- | --- | --- | --- | --- |
| 3 | 15 | 15 | 0 | 11.1 ms | 0.17 s |
| 5 | 45 | 45 | 0 | 22.0 ms | 0.99 s |
| 7 | 91 | 91 | 0 | 64.4 ms | 5.86 s |

Mechanism count is `distance * rounds` data flips plus `(distance - 1) * rounds` measurement flips. Every mechanism at every measured distance produced a unique signature and none was empty, so merging and discarding never fire on the repetition code — both paths still need coverage from a hand-built model and a second code.

`lower_dynamic_program(program, (rounds,), max_dynamic_measurements=rounds * len(checks))` is the sizing that matches the emitted measurement count. Lowering dominates the per-mechanism cost (~9.5 ms of 11.1 ms at `d = 3`); execution is 6–7 ms.

Verified signatures at `d = 3`, `rounds = 3`: 8 detectors, one observable, 2 checks, so detector index `r * 2 + c` for the round detectors and `6 + c` for the terminal ones.

| mechanism | detector signature | observable |
| --- | --- | --- |
| data flip, round 0, wire 0 | `D0` | 1 |
| data flip, round 0, wire 1 | `D0 D1` | 1 |
| data flip, round 2, wire 0 | `D4` | 1 |
| measurement flip, round 0, ancilla 3 (check 0) | `D0 D2` | 0 |
| measurement flip, round 2, ancilla 3 (check 0) | `D4 D6` | 0 |

The measurement rows are the ones worth reading twice. A flipped syndrome measurement is compared against **both** of its neighbours, so it flips its own round's detector and the next round's detector — and it flips no observable. At the last round the second detector is that check's terminal boundary. A data flip always flips the observable, because the memory circuit applies no correction.

Cross-check cost: a shot costs one lowering plus one execution, so ~15 ms at `d = 3` and ~22 ms at `d = 5`. The plan's budget is `d = 3`: 1000 shots (~15 s) and `d = 5`: 500 shots (~11 s), plus the exhaustive pairwise sweep at `d = 3`, which is `C(15, 2) = 105` executions (~1.2 s).

---

## File Structure

| File | Responsibility |
| --- | --- |
| `flagquantum/qec/noise.py` (modify) | Add `PhenomenologicalNoise`; leave the frozen profile untouched |
| `flagquantum/qec/dem.py` (create) | `DemError`, `DetectorErrorModel`, exact rates, sampling, stim text, the forced-error signature engine, and `from_memory_circuit` |
| `flagquantum/qec/__init__.py` (modify) | Publish the new names beside the Stage 1 and frozen ones |
| `flagquantum/qec/IMPLEMENTATION.md` (modify) | Describe the DEM layer and its boundary |
| `tests/qec/test_phenomenological_noise.py` (create) | Unit coverage for `PhenomenologicalNoise` |
| `tests/qec/test_dem_records.py` (create) | Unit coverage for `DemError` and `DetectorErrorModel` validation |
| `tests/qec/test_dem_rates.py` (create) | Unit coverage for the parity matrices and the exact marginal rates |
| `tests/qec/test_dem_sampling.py` (create) | Unit coverage for seeded `dem_sampling` |
| `tests/qec/test_dem_stim_text.py` (create) | Unit coverage for the text round trip |
| `tests/qec/test_dem_signatures.py` (create) | Coverage for the forced-error engine against hand-derived signatures |
| `tests/qec/test_dem_from_memory_circuit.py` (create) | Coverage for `from_memory_circuit`, including non-repetition-shaped and index-swapped codes |
| `tests/qec/test_dem_cross_check.py` (create) | Integration coverage: pairwise linearity and sampled-rate agreement |

---

### Task 1: Phenomenological noise record

**Files:**
- Modify: `flagquantum/qec/noise.py`
- Test: `tests/qec/test_phenomenological_noise.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. Reuses the module's existing `_probability` validator at `flagquantum/qec/noise.py:15`.
- Produces: `PhenomenologicalNoise`, a frozen dataclass with fields `data_flip: float = 0.0` and `measurement_flip: float = 0.0`. `data_flip` is the independent X probability applied to **every data wire at the start of every syndrome round**; `measurement_flip` is the independent flip probability applied to **every syndrome measurement of every check in every round**. It is a description of noise locations, not a `NoiseModel`: it deliberately has no `to_noise_model()`, because the round-boundary data location has no matching hook in the executor's `NoiseModel` grammar.

- [ ] **Step 1: Write the failing test**

Create `tests/qec/test_phenomenological_noise.py`:

```python
"""Unit coverage for the phenomenological noise record."""

from __future__ import annotations

import dataclasses

import pytest

from flagquantum.qec.noise import PhenomenologicalNoise, RepetitionNoiseProfile

pytestmark = pytest.mark.unit


def test_defaults_are_noiseless() -> None:
    noise = PhenomenologicalNoise()
    assert noise.data_flip == 0.0
    assert noise.measurement_flip == 0.0


def test_fields_round_trip() -> None:
    noise = PhenomenologicalNoise(data_flip=0.05, measurement_flip=0.02)
    assert noise.data_flip == 0.05
    assert noise.measurement_flip == 0.02


def test_record_is_frozen() -> None:
    noise = PhenomenologicalNoise(data_flip=0.05)
    with pytest.raises(dataclasses.FrozenInstanceError):
        noise.data_flip = 0.1  # type: ignore[misc]


@pytest.mark.parametrize("field", ["data_flip", "measurement_flip"])
def test_probability_bounds_are_enforced(field: str) -> None:
    with pytest.raises(ValueError):
        PhenomenologicalNoise(**{field: -0.01})
    with pytest.raises(ValueError):
        PhenomenologicalNoise(**{field: 1.01})


@pytest.mark.parametrize("field", ["data_flip", "measurement_flip"])
def test_bool_is_rejected_as_a_probability(field: str) -> None:
    with pytest.raises(TypeError):
        PhenomenologicalNoise(**{field: True})


def test_boundary_values_are_accepted() -> None:
    assert PhenomenologicalNoise(data_flip=0.0, measurement_flip=1.0).measurement_flip == 1.0


def test_frozen_profile_is_unchanged() -> None:
    """The additive rule: the frozen profile keeps its fields and defaults."""

    profile = RepetitionNoiseProfile()
    assert profile.data_bit_flip_probability == 0.0
    assert profile.syndrome_readout_error_probability == 0.0
    assert profile.final_readout_error_probability == 0.0
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
python -m pytest tests/qec/test_phenomenological_noise.py -q
```

Expected: collection error, `ImportError: cannot import name 'PhenomenologicalNoise'`.

- [ ] **Step 3: Implement the record**

Append to `flagquantum/qec/noise.py`, below `RepetitionNoiseProfile` and above `run_repetition_memory_noise_sweep`:

```python
@dataclass(frozen=True)
class PhenomenologicalNoise:
    """Independent data and measurement flips at fixed circuit locations.

    ``data_flip`` applies to every data wire at the start of every syndrome
    round, before that round's parity-check CNOTs. ``measurement_flip`` applies
    to every syndrome measurement of every check in every round. Both are
    independent per location per round.

    This is a description of noise locations, not a ``NoiseModel``. The
    round-boundary data location has no equivalent in the executor's
    instruction-ordered noise grammar, so no ``to_noise_model`` is offered and
    the DEM's detector-rate cross-check drives this record through forced
    injection instead.
    """

    data_flip: float = 0.0
    measurement_flip: float = 0.0

    def __post_init__(self) -> None:
        for name in ("data_flip", "measurement_flip"):
            object.__setattr__(self, name, _probability(getattr(self, name), name=name))
```

- [ ] **Step 4: Run the test and confirm it passes**

```bash
python -m pytest tests/qec/test_phenomenological_noise.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Confirm the frozen profile's own tests still pass**

```bash
python -m pytest tests/qec -q
```

Expected: the Stage 1 count plus the new tests, all passing.

- [ ] **Step 6: Commit**

```bash
git add flagquantum/qec/noise.py tests/qec/test_phenomenological_noise.py
git commit -m "feat: describe phenomenological data and measurement flips"
```

---

### Task 2: DemError and DetectorErrorModel records

**Files:**
- Create: `flagquantum/qec/dem.py`
- Test: `tests/qec/test_dem_records.py`

**Interfaces:**
- Consumes: `flagquantum.qec.pauli` is **not** needed here; these records are integer-indexed.
- Produces:
  - `DemError(probability: float, detectors: tuple[int, ...], observables: tuple[int, ...])` — a frozen dataclass. `detectors` and `observables` are sorted ascending without duplicates, and at least one of them is non-empty.
  - `DetectorErrorModel(num_detectors: int, num_observables: int, errors: tuple[DemError, ...])` — a frozen dataclass whose `errors` are stored in a canonical order, with a `num_errors` property.

- [ ] **Step 1: Write the failing test**

Create `tests/qec/test_dem_records.py`:

```python
"""Unit coverage for the detector-error-model records."""

from __future__ import annotations

import pytest

from flagquantum.qec.dem import DemError, DetectorErrorModel

pytestmark = pytest.mark.unit


def test_error_stores_its_signature() -> None:
    error = DemError(probability=0.05, detectors=(0, 2), observables=(1,))
    assert error.probability == 0.05
    assert error.detectors == (0, 2)
    assert error.observables == (1,)


def test_error_normalizes_its_signature() -> None:
    error = DemError(probability=0.05, detectors=(2, 0, 2), observables=(1, 1))
    assert error.detectors == (0, 2)
    assert error.observables == (1,)


def test_error_rejects_an_empty_signature() -> None:
    with pytest.raises(ValueError, match="must flip at least one"):
        DemError(probability=0.05, detectors=(), observables=())


@pytest.mark.parametrize("probability", [-0.1, 1.5])
def test_error_rejects_an_out_of_range_probability(probability: float) -> None:
    with pytest.raises(ValueError):
        DemError(probability=probability, detectors=(0,), observables=())


def test_error_rejects_a_bool_probability() -> None:
    with pytest.raises(TypeError):
        DemError(probability=True, detectors=(0,), observables=())


def test_error_rejects_a_negative_index() -> None:
    with pytest.raises(ValueError):
        DemError(probability=0.05, detectors=(-1,), observables=())


def test_model_stores_its_shape() -> None:
    model = DetectorErrorModel(
        num_detectors=4,
        num_observables=1,
        errors=(DemError(probability=0.05, detectors=(0,), observables=(0,)),),
    )
    assert model.num_detectors == 4
    assert model.num_observables == 1
    assert model.num_errors == 1


def test_model_accepts_no_errors() -> None:
    model = DetectorErrorModel(num_detectors=2, num_observables=1, errors=())
    assert model.num_errors == 0


def test_model_rejects_an_out_of_range_detector() -> None:
    with pytest.raises(ValueError, match="detector index"):
        DetectorErrorModel(
            num_detectors=2,
            num_observables=1,
            errors=(DemError(probability=0.05, detectors=(2,), observables=()),),
        )


def test_model_rejects_an_out_of_range_observable() -> None:
    with pytest.raises(ValueError, match="observable index"):
        DetectorErrorModel(
            num_detectors=2,
            num_observables=1,
            errors=(DemError(probability=0.05, detectors=(0,), observables=(1,)),),
        )


def test_model_orders_errors_canonically() -> None:
    """Equal models built in different orders compare equal."""

    first = DemError(probability=0.05, detectors=(1,), observables=())
    second = DemError(probability=0.02, detectors=(0,), observables=(0,))
    forward = DetectorErrorModel(num_detectors=2, num_observables=1, errors=(first, second))
    backward = DetectorErrorModel(num_detectors=2, num_observables=1, errors=(second, first))
    assert forward == backward
    assert forward.errors == (second, first)


def test_model_requires_a_positive_detector_count() -> None:
    with pytest.raises(ValueError, match="at least one detector"):
        DetectorErrorModel(num_detectors=0, num_observables=1, errors=())


def test_model_allows_a_model_without_observables() -> None:
    model = DetectorErrorModel(num_detectors=1, num_observables=0, errors=())
    assert model.num_observables == 0
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
python -m pytest tests/qec/test_dem_records.py -q
```

Expected: collection error, `ModuleNotFoundError: No module named 'flagquantum.qec.dem'`.

- [ ] **Step 3: Implement the records**

Create `flagquantum/qec/dem.py`:

```python
"""Exact detector error models for code-independent memory experiments.

A detector error model names every independent physical error mechanism by the
detectors and logical observables it flips. Construction is exact and does not
sample: the reference gate set is Clifford and every configured channel is a
Pauli channel, so forcing one mechanism through the circuit yields a
deterministic signature.

The model is defined for Pauli noise only. It is built on the memory circuit
*without* in-circuit feedback, because the model describes the physical
noise-to-detection mapping that a decoder inverts; the compiled feedback layer
is what a decoder replaces.

Detector rates the model predicts are compared against rates sampled from the
circuit simulator by the tests, not by this module. The comparison shares the
injection helper with construction, so it does not independently re-derive the
signatures; what it tests is that mechanisms compose by XOR in the simulator and
that merging identical signatures yields the right marginals.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Integral, Real

_DETECTOR_PREFIX = "D"
_OBSERVABLE_PREFIX = "L"


def _probability(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real probability")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return normalized


def _count(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


def _normalized_indices(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    indices: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must contain integer indices")
        index = int(value)
        if index < 0:
            raise ValueError(f"{name} must contain non-negative indices")
        indices.append(index)
    return tuple(sorted(set(indices)))


@dataclass(frozen=True)
class DemError:
    """One independent error mechanism and the signature it flips."""

    probability: float
    detectors: tuple[int, ...] = ()
    observables: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "probability", _probability(self.probability, name="error probability")
        )
        object.__setattr__(
            self,
            "detectors",
            _normalized_indices(self.detectors, name="error detectors"),
        )
        object.__setattr__(
            self,
            "observables",
            _normalized_indices(self.observables, name="error observables"),
        )
        if not self.detectors and not self.observables:
            raise ValueError("an error mechanism must flip at least one detector or observable")


@dataclass(frozen=True)
class DetectorErrorModel:
    """A detector error model over a fixed number of detectors and observables."""

    num_detectors: int
    num_observables: int
    errors: tuple[DemError, ...] = ()

    def __post_init__(self) -> None:
        detectors = _count(self.num_detectors, name="num_detectors")
        observables = _count(self.num_observables, name="num_observables")
        if detectors < 1:
            raise ValueError("a detector error model requires at least one detector")
        object.__setattr__(self, "num_detectors", detectors)
        object.__setattr__(self, "num_observables", observables)

        for error in self.errors:
            if not isinstance(error, DemError):
                raise TypeError("errors must contain DemError records")
        object.__setattr__(
            self,
            "errors",
            tuple(
                sorted(
                    self.errors,
                    key=lambda error: (error.detectors, error.observables, error.probability),
                )
            ),
        )
        for error in self.errors:
            for index in error.detectors:
                if index >= detectors:
                    raise ValueError(
                        f"detector index {index} is outside the model shape"
                    )
            for index in error.observables:
                if index >= observables:
                    raise ValueError(
                        f"observable index {index} is outside the model shape"
                    )

    @property
    def num_errors(self) -> int:
        """Number of independent error mechanisms in the model."""

        return len(self.errors)


__all__ = ("DemError", "DetectorErrorModel")
```

- [ ] **Step 4: Run the test and confirm it passes**

```bash
python -m pytest tests/qec/test_dem_records.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Run the quality gates on the new module**

```bash
black --check flagquantum/qec/dem.py tests/qec/test_dem_records.py
ruff check flagquantum/qec/dem.py tests/qec/test_dem_records.py
mypy --strict --python-version 3.12 --ignore-missing-imports flagquantum
python tools/check_architecture.py
```

Expected: all clean. `mypy --strict` will require the `_normalized_indices` overload-free signature above to satisfy the call sites; adjust annotations rather than adding `# type: ignore`.

- [ ] **Step 6: Commit**

```bash
git add flagquantum/qec/dem.py tests/qec/test_dem_records.py
git commit -m "feat: add detector error model records"
```

---

### Task 3: Parity matrices and exact marginal rates

**Files:**
- Modify: `flagquantum/qec/dem.py`
- Test: `tests/qec/test_dem_rates.py`

**Interfaces:**
- Consumes: `DetectorErrorModel` from Task 2.
- Produces, on `DetectorErrorModel`:
  - `detector_error_matrix() -> torch.Tensor` — `(num_detectors, num_errors)` `torch.int8`, entry `[d, e] = 1` when error `e` flips detector `d`.
  - `observables_flips_matrix() -> torch.Tensor` — `(num_observables, num_errors)` `torch.int8`.
  - `detector_rates() -> torch.Tensor` — `(num_detectors,)` `torch.float64`, the exact marginal flip probability of each detector.
  - `observable_rates() -> torch.Tensor` — `(num_observables,)` `torch.float64`.

- [ ] **Step 1: Write the failing test**

Create `tests/qec/test_dem_rates.py`:

```python
"""Unit coverage for parity matrices and exact marginal rates."""

from __future__ import annotations

import pytest
import torch

from flagquantum.qec.dem import DemError, DetectorErrorModel

pytestmark = pytest.mark.unit


def _model(*errors: DemError, detectors: int = 4, observables: int = 1):
    return DetectorErrorModel(
        num_detectors=detectors, num_observables=observables, errors=errors
    )


def test_detector_matrix_orientation_and_values() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0, 1), observables=(0,)),
        DemError(probability=0.2, detectors=(1,), observables=()),
        detectors=3,
    )
    matrix = model.detector_error_matrix()
    assert matrix.shape == (3, 2)
    assert matrix.dtype == torch.int8
    assert matrix.tolist() == [[1, 0], [1, 1], [0, 0]]


def test_observable_matrix_orientation_and_values() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=(0,)),
        DemError(probability=0.2, detectors=(1,), observables=()),
        detectors=3,
    )
    matrix = model.observables_flips_matrix()
    assert matrix.shape == (1, 2)
    assert matrix.dtype == torch.int8
    assert matrix.tolist() == [[1, 0]]


def test_empty_model_has_empty_matrices() -> None:
    model = DetectorErrorModel(num_detectors=3, num_observables=2, errors=())
    assert model.detector_error_matrix().shape == (3, 0)
    assert model.observables_flips_matrix().shape == (2, 0)


def test_single_error_rates_are_its_own_probabilities() -> None:
    model = _model(
        DemError(probability=0.25, detectors=(0, 1), observables=(0,)),
        detectors=2,
    )
    assert model.detector_rates().tolist() == pytest.approx([0.25, 0.25])
    assert model.observable_rates().tolist() == pytest.approx([0.25])


def test_untouched_detector_has_zero_rate() -> None:
    model = _model(DemError(probability=0.3, detectors=(0,), observables=()), detectors=2)
    assert model.detector_rates().tolist() == pytest.approx([0.3, 0.0])


def test_shared_detector_combines_by_parity_not_by_sum() -> None:
    """Two mechanisms on one detector flip it when exactly one fires."""

    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=()),
        DemError(probability=0.2, detectors=(0,), observables=()),
        detectors=1,
    )
    expected = 0.1 * 0.8 + 0.2 * 0.9
    assert model.detector_rates().tolist() == pytest.approx([expected])


def test_three_mechanisms_use_the_closed_form() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=()),
        DemError(probability=0.2, detectors=(0,), observables=()),
        DemError(probability=0.4, detectors=(0,), observables=()),
        detectors=1,
    )
    expected = (1 - (0.8 * 0.6 * 0.2)) / 2
    assert model.detector_rates().tolist() == pytest.approx([expected])


def test_rates_are_independent_across_detectors() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0, 1), observables=(0,)),
        DemError(probability=0.2, detectors=(1, 2), observables=(0,)),
        detectors=3,
    )
    rates = model.detector_rates().tolist()
    assert rates[0] == pytest.approx(0.1)
    assert rates[1] == pytest.approx(0.1 * 0.8 + 0.2 * 0.9)
    assert rates[2] == pytest.approx(0.2)


def test_a_certain_error_yields_certain_rates() -> None:
    model = _model(DemError(probability=1.0, detectors=(0,), observables=(0,)), detectors=1)
    assert model.detector_rates().tolist() == pytest.approx([1.0])
    assert model.observable_rates().tolist() == pytest.approx([1.0])


def test_rates_are_float64() -> None:
    model = _model(DemError(probability=0.1, detectors=(0,), observables=()), detectors=1)
    assert model.detector_rates().dtype == torch.float64
    assert model.observable_rates().dtype == torch.float64


def test_observable_rate_ignores_detector_only_errors() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=()),
        DemError(probability=0.2, detectors=(1,), observables=(0,)),
        detectors=2,
    )
    assert model.observable_rates().tolist() == pytest.approx([0.2])
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
python -m pytest tests/qec/test_dem_rates.py -q
```

Expected: `AttributeError: 'DetectorErrorModel' object has no attribute 'detector_error_matrix'`.

- [ ] **Step 3: Implement the matrices and rates**

Add to `flagquantum/qec/dem.py`, importing torch at the top beside the other imports:

```python
import torch
```

and appending these methods to `DetectorErrorModel`, above `num_errors`:

```python
    def detector_error_matrix(self) -> torch.Tensor:
        """Return the ``(num_detectors, num_errors)`` parity matrix.

        Entry ``[d, e]`` is one when error ``e`` flips detector ``d``. The
        orientation matches the stim ecosystem's detector error matrix.
        """

        matrix = torch.zeros(
            (self.num_detectors, self.num_errors), dtype=torch.int8
        )
        for column, error in enumerate(self.errors):
            for index in error.detectors:
                matrix[index, column] = 1
        return matrix

    def observables_flips_matrix(self) -> torch.Tensor:
        """Return the ``(num_observables, num_errors)`` parity matrix.

        Entry ``[o, e]`` is one when error ``e`` flips observable ``o``.
        """

        matrix = torch.zeros(
            (self.num_observables, self.num_errors), dtype=torch.int8
        )
        for column, error in enumerate(self.errors):
            for index in error.observables:
                matrix[index, column] = 1
        return matrix

    def _marginal_rates(
        self, count: int, select: Callable[[DemError], tuple[int, ...]]
    ) -> torch.Tensor:
        rates = torch.zeros((count,), dtype=torch.float64)
        if count == 0:
            return rates
        complements = torch.ones((count,), dtype=torch.float64)
        for error in self.errors:
            factor = 1.0 - 2.0 * error.probability
            for index in select(error):
                complements[index] *= factor
        return (1.0 - complements) / 2.0

    def detector_rates(self) -> torch.Tensor:
        """Return the exact marginal flip probability of every detector."""

        return self._marginal_rates(self.num_detectors, lambda error: error.detectors)

    def observable_rates(self) -> torch.Tensor:
        """Return the exact marginal flip probability of every observable."""

        return self._marginal_rates(self.num_observables, lambda error: error.observables)
```

- [ ] **Step 4: Run the test and confirm it passes**

```bash
python -m pytest tests/qec/test_dem_rates.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add flagquantum/qec/dem.py tests/qec/test_dem_rates.py
git commit -m "feat: derive DEM parity matrices and exact marginal rates"
```

---

### Task 4: Seeded sampling

**Files:**
- Modify: `flagquantum/qec/dem.py`
- Test: `tests/qec/test_dem_sampling.py`

**Interfaces:**
- Consumes: `DetectorErrorModel` from Task 2, its exact rates from Task 3.
- Produces:
  - `DemSample(detectors: torch.Tensor, observables: torch.Tensor)` — a frozen dataclass; `detectors` is `(shots, num_detectors)` `torch.int8`, `observables` is `(shots, num_observables)` `torch.int8`, and `shots` is a property.
  - `DetectorErrorModel.dem_sampling(*, shots: int, seed: int | None = None) -> DemSample`.

- [ ] **Step 1: Write the failing test**

Create `tests/qec/test_dem_sampling.py`:

```python
"""Unit coverage for seeded detector error model sampling."""

from __future__ import annotations

import pytest
import torch

from flagquantum.qec.dem import DemError, DetectorErrorModel

pytestmark = pytest.mark.unit


def _model(*errors: DemError, detectors: int = 4, observables: int = 1):
    return DetectorErrorModel(
        num_detectors=detectors, num_observables=observables, errors=errors
    )


def test_sample_shapes_and_dtypes() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0, 1), observables=(0,)), detectors=4
    )
    sample = model.dem_sampling(shots=16, seed=0)
    assert sample.detectors.shape == (16, 4)
    assert sample.observables.shape == (16, 1)
    assert sample.detectors.dtype == torch.int8
    assert sample.observables.dtype == torch.int8
    assert sample.shots == 16


def test_sampling_is_reproducible_for_a_seed() -> None:
    model = _model(DemError(probability=0.3, detectors=(0,), observables=(0,)), detectors=2)
    first = model.dem_sampling(shots=64, seed=7)
    second = model.dem_sampling(shots=64, seed=7)
    assert torch.equal(first.detectors, second.detectors)
    assert torch.equal(first.observables, second.observables)


def test_different_seeds_differ() -> None:
    model = _model(DemError(probability=0.3, detectors=(0,), observables=(0,)), detectors=2)
    assert not torch.equal(
        model.dem_sampling(shots=64, seed=1).detectors,
        model.dem_sampling(shots=64, seed=2).detectors,
    )


def test_an_error_free_model_never_flips() -> None:
    model = DetectorErrorModel(num_detectors=3, num_observables=1, errors=())
    sample = model.dem_sampling(shots=8, seed=0)
    assert sample.detectors.shape == (8, 3)
    assert int(sample.detectors.sum()) == 0
    assert int(sample.observables.sum()) == 0


def test_a_certain_error_always_flips() -> None:
    model = _model(DemError(probability=1.0, detectors=(0,), observables=(0,)), detectors=2)
    sample = model.dem_sampling(shots=8, seed=0)
    assert int(sample.detectors[:, 0].sum()) == 8
    assert int(sample.observables[:, 0].sum()) == 8


def test_a_shared_error_flips_both_detectors_together() -> None:
    """One mechanism that flips two detectors flips them together."""

    model = _model(DemError(probability=0.5, detectors=(0, 1), observables=()), detectors=2)
    sample = model.dem_sampling(shots=256, seed=3)
    assert torch.equal(sample.detectors[:, 0], sample.detectors[:, 1])


def test_sampled_rates_track_the_exact_rates() -> None:
    model = _model(
        DemError(probability=0.1, detectors=(0,), observables=(0,)),
        DemError(probability=0.2, detectors=(0, 1), observables=(0,)),
        detectors=2,
    )
    sample = model.dem_sampling(shots=20_000, seed=11)
    sampled = sample.detectors.to(torch.float64).mean(dim=0)
    assert sampled.tolist() == pytest.approx(model.detector_rates().tolist(), abs=0.01)
    sampled_observable = sample.observables.to(torch.float64).mean(dim=0)
    assert sampled_observable.tolist() == pytest.approx(
        model.observable_rates().tolist(), abs=0.01
    )


@pytest.mark.parametrize("shots", [0, -1])
def test_shots_must_be_positive(shots: int) -> None:
    model = _model(DemError(probability=0.1, detectors=(0,), observables=()), detectors=1)
    with pytest.raises(ValueError, match="shots must be a positive integer"):
        model.dem_sampling(shots=shots, seed=0)


def test_bool_shots_are_rejected() -> None:
    model = _model(DemError(probability=0.1, detectors=(0,), observables=()), detectors=1)
    with pytest.raises(TypeError, match="shots must be a positive integer"):
        model.dem_sampling(shots=True, seed=0)
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
python -m pytest tests/qec/test_dem_sampling.py -q
```

Expected: `ImportError: cannot import name 'DemSample'`.

- [ ] **Step 3: Implement the sampler**

Add to `flagquantum/qec/dem.py`: a `DemSample` record after `DemError`, and a `dem_sampling` method on `DetectorErrorModel` after `observable_rates`. Sampling draws one Bernoulli per mechanism per shot and XORs the signatures of the mechanisms that fired, which is exactly the model's independence assumption.

```python
@dataclass(frozen=True)
class DemSample:
    """Sampled detector and observable flips from a detector error model."""

    detectors: torch.Tensor
    observables: torch.Tensor

    @property
    def shots(self) -> int:
        """Number of sampled shots."""

        return int(self.detectors.shape[0])
```

```python
    def dem_sampling(self, *, shots: int, seed: int | None = None) -> DemSample:
        """Sample ``shots`` shots by XORing the signatures that fired.

        Every mechanism is drawn independently per shot with its own
        probability. This is the model's own arithmetic, not the circuit
        simulator; the circuit-simulator comparison lives in the tests.
        """

        if isinstance(shots, bool) or not isinstance(shots, Integral):
            raise TypeError("shots must be a positive integer")
        if shots <= 0:
            raise ValueError("shots must be a positive integer")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
            raise TypeError("seed must be an integer or None")

        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))

        detector_bits = torch.zeros((shots, self.num_detectors), dtype=torch.int8)
        observable_bits = torch.zeros((shots, self.num_observables), dtype=torch.int8)
        if not self.errors:
            return DemSample(detectors=detector_bits, observables=observable_bits)

        probabilities = torch.tensor(
            [error.probability for error in self.errors], dtype=torch.float64
        )
        fired = torch.rand((shots, self.num_errors), generator=generator, dtype=torch.float64)
        fired = fired < probabilities
        for column, error in enumerate(self.errors):
            active = fired[:, column]
            for index in error.detectors:
                detector_bits[:, index] ^= active.to(torch.int8)
            for index in error.observables:
                observable_bits[:, index] ^= active.to(torch.int8)
        return DemSample(detectors=detector_bits, observables=observable_bits)
```

Note the two-part `shots` guard: `bool` is an `Integral` subclass, so it has to be rejected before the type check, and the sign check has to come after, because the test asserts `TypeError` for `True` and `ValueError` for `0` and `-1`.

- [ ] **Step 4: Run the test and confirm it passes**

```bash
python -m pytest tests/qec/test_dem_sampling.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add flagquantum/qec/dem.py tests/qec/test_dem_sampling.py
git commit -m "feat: sample a detector error model from its mechanisms"
```

---

### Task 5: stim text interchange

**Files:**
- Modify: `flagquantum/qec/dem.py`
- Test: `tests/qec/test_dem_stim_text.py`

**Interfaces:**
- Consumes: `DemError`, `DetectorErrorModel` from Task 2.
- Produces, on `DetectorErrorModel`:
  - `to_stim_text() -> str`
  - `from_stim_text(text: str) -> DetectorErrorModel` (classmethod)

Format, pinned exactly:

```
error(0.05) D0 D1 L0
detector D0
detector D1
logical_observable L0
```

One `error(...)` line per mechanism, detectors ascending and then observables ascending within the line, probabilities formatted with `repr(float)` so the value round-trips. One `detector D<i>` declaration per detector and one `logical_observable L<i>` per observable, so a model whose trailing detectors are touched by no error keeps its shape. Lines end with `\n`, including the last.

`from_stim_text` reads the shape back from the declarations, not from the largest index an error mentions. It accepts `error(...)`, `detector D<i>`, `detector(...) D<i>`, and `logical_observable L<i>`, and rejects everything else — including `shift_detectors`, `repeat`, and coordinate syntax — with a stated reason, so an unsupported construct fails closed instead of being silently dropped. stim is **not** added as a dependency; this is a text format, not an import.

- [ ] **Step 1: Write the failing test**

Create `tests/qec/test_dem_stim_text.py`:

```python
"""Unit coverage for the stim text interchange."""

from __future__ import annotations

import pytest

from flagquantum.qec.dem import DemError, DetectorErrorModel

pytestmark = pytest.mark.unit


def _model(*errors: DemError, detectors: int = 3, observables: int = 1):
    return DetectorErrorModel(
        num_detectors=detectors, num_observables=observables, errors=errors
    )


def test_exact_text_for_a_hand_built_model() -> None:
    model = _model(
        DemError(probability=0.05, detectors=(0, 1), observables=(0,)),
        DemError(probability=0.125, detectors=(2,), observables=()),
    )
    assert model.to_stim_text() == (
        "error(0.05) D0 D1 L0\n"
        "error(0.125) D2\n"
        "detector D0\n"
        "detector D1\n"
        "detector D2\n"
        "logical_observable L0\n"
    )


def test_error_free_model_still_declares_its_shape() -> None:
    model = DetectorErrorModel(num_detectors=2, num_observables=1, errors=())
    assert model.to_stim_text() == (
        "detector D0\ndetector D1\nlogical_observable L0\n"
    )


def test_round_trip_preserves_the_model() -> None:
    model = _model(
        DemError(probability=0.05, detectors=(0, 1), observables=(0,)),
        DemError(probability=1 / 3, detectors=(2,), observables=(0,)),
        detectors=3,
    )
    assert DetectorErrorModel.from_stim_text(model.to_stim_text()) == model


def test_round_trip_preserves_a_shape_no_error_touches() -> None:
    """The declaration lines are what make this lossless."""

    model = DetectorErrorModel(num_detectors=5, num_observables=2, errors=())
    restored = DetectorErrorModel.from_stim_text(model.to_stim_text())
    assert restored.num_detectors == 5
    assert restored.num_observables == 2


def test_round_trip_is_stable() -> None:
    model = _model(DemError(probability=1 / 3, detectors=(0,), observables=(0,)), detectors=1)
    once = model.to_stim_text()
    assert DetectorErrorModel.from_stim_text(once).to_stim_text() == once


def test_parses_coordinate_declarations() -> None:
    text = (
        "error(0.1) D0 D1 L0\n"
        "error(0.2) D2\n"
        "detector(1, 2) D0\n"
        "detector D1\n"
        "detector D2\n"
        "logical_observable L0\n"
    )
    model = DetectorErrorModel.from_stim_text(text)
    assert model.num_detectors == 3
    assert model.num_observables == 1
    assert model.num_errors == 2


def test_rejects_an_unsupported_instruction() -> None:
    with pytest.raises(ValueError, match="shift_detectors"):
        DetectorErrorModel.from_stim_text("shift_detectors 1\n")


def test_rejects_a_repeat_block() -> None:
    with pytest.raises(ValueError, match="repeat"):
        DetectorErrorModel.from_stim_text("repeat 2 {\n error(0.1) D0\n}\n")


def test_rejects_an_error_with_no_effect() -> None:
    with pytest.raises(ValueError, match="must flip at least one"):
        DetectorErrorModel.from_stim_text("error(0.1)\n")


def test_rejects_a_probability_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError):
        DetectorErrorModel.from_stim_text("error(1.5) D0\n")


def test_rejects_a_malformed_target() -> None:
    with pytest.raises(ValueError, match="D or L"):
        DetectorErrorModel.from_stim_text("error(0.1) X0\n")


def test_rejects_a_model_with_no_detectors() -> None:
    with pytest.raises(ValueError, match="at least one detector"):
        DetectorErrorModel.from_stim_text("logical_observable L0\n")


def test_rejects_a_bare_string_without_declarations() -> None:
    with pytest.raises(ValueError, match="at least one detector"):
        DetectorErrorModel.from_stim_text("error(0.1) D0\n")
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
python -m pytest tests/qec/test_dem_stim_text.py -q
```

Expected: `AttributeError: 'DetectorErrorModel' object has no attribute 'to_stim_text'`.

- [ ] **Step 3: Implement the interchange**

Add a private module-level parser and two methods. Parsing rules:

- Split on newlines, strip, skip empty lines.
- `detector D<i>` → record detector declaration `i`.
- `detector(<anything>) D<i>` → accept and ignore the coordinates, record `i`.
- `logical_observable L<i>` → record observable declaration `i`.
- `error(<p>) <targets...>` → parse `p` with `float`, `D<i>` into detectors, `L<i>` into observables, anything else raises `ValueError` naming the offending token as neither `D` nor `L`.
- Any other leading keyword raises `ValueError` naming it; `repeat` and `shift_detectors` get their own message because they are the two a reader is most likely to expect to work.

`num_detectors` and `num_observables` come from the declaration lines and nowhere else: the text must declare one `detector` per detector and one `logical_observable` per observable. Shape is never inferred from the largest index an error mentions, because that would silently accept a text that had lost its trailing detectors. A text with no `detector` declaration therefore fails the constructor's "at least one detector" rule.

Sketch to adapt:

```python
def _parse_error_line(line: str) -> DemError:
    body = line[len("error(") :]
    closing = body.find(")")
    if closing < 0:
        raise ValueError("error line must close its probability in parentheses")
    try:
        probability = float(body[:closing])
    except ValueError as error:
        raise ValueError("error line must state a numeric probability") from error
    detectors: list[int] = []
    observables: list[int] = []
    remainder = body[closing + 1 :].split()
    for token in remainder:
        if token.startswith(_DETECTOR_PREFIX) and token[1:].isdigit():
            detectors.append(int(token[1:]))
        elif token.startswith(_OBSERVABLE_PREFIX) and token[1:].isdigit():
            observables.append(int(token[1:]))
        else:
            raise ValueError(f"error targets must be D or L indices, not {token!r}")
    if not detectors and not observables:
        raise ValueError("an error mechanism must flip at least one detector or observable")
    return DemError(
        probability=probability, detectors=tuple(detectors), observables=tuple(observables)
    )
```

- [ ] **Step 4: Run the test and confirm it passes**

```bash
python -m pytest tests/qec/test_dem_stim_text.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Verify the emitted text against the real stim package (developer-time only)**

This is **not** a committed test and must not add stim as a dependency. Write the throwaway check to `.qec_stim_check.py` in the worktree (the leading dot keeps it out of the tracked tree; delete it when done), sync it, and run it in the container with stim installed from the Tsinghua mirror, because the hosts cannot reach `pypi.org`:

```python
# .qec_stim_check.py — throwaway
import stim

from flagquantum.qec.circuit import build_memory_circuit
from flagquantum.qec.codes import RepetitionCode
from flagquantum.qec.dem import DetectorErrorModel
from flagquantum.qec.noise import PhenomenologicalNoise

built = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
model = DetectorErrorModel.from_memory_circuit(
    built, noise=PhenomenologicalNoise(data_flip=0.05, measurement_flip=0.05)
)
parsed = stim.DetectorErrorModel(model.to_stim_text())
print("stim ", parsed.num_detectors, parsed.num_observables, parsed.num_errors)
print("model", model.num_detectors, model.num_observables, model.num_errors)
```

```bash
rsync -az .qec_stim_check.py jp-a800-172:/root/liuwei/fq-qec-run/
ssh jp-a800-172 'docker run --rm --name fq-qec-claude-stim \
  -v /root/liuwei/fq-qec-run:/w -w /w -e PYTHONPATH=/w --entrypoint bash \
  tovx/flagquantum:0.2.0-dev -lc \
  "python -m pip install -q -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple stim \
   && python .qec_stim_check.py"'
```

Expected: the two triples agree. If stim's parser disagrees, fix the emitted text — the text format is the contract, and `to_stim_text` is the half that must move. This cross-check is the only genuinely independent verification of the model's construction available in this stage; record its result in the task's commit message or the pull request description. Then delete `.qec_stim_check.py`, and note that the next `rsync --delete` removes the remote copy.

- [ ] **Step 6: Commit**

```bash
git add flagquantum/qec/dem.py tests/qec/test_dem_stim_text.py
git commit -m "feat: interchange detector error models as stim text"
```

---

### Task 6: Forced-error signature engine

**Files:**
- Modify: `flagquantum/qec/dem.py`
- Test: `tests/qec/test_dem_signatures.py`

**Interfaces:**
- Consumes: `MemoryCircuit` and its layouts from `flagquantum.qec.circuit`; `PhenomenologicalNoise` from Task 1; the private compiler path `flagquantum.compiler._hybrid` (`INDEX`, `capture_source`, `lower_dynamic_program`); `flagquantum.runtime.dynamic.hybrid_session.execute_hybrid_dynamic_session`.
- Produces three module-private functions in `dem.py`, importable by tests but absent from `__all__`:
  - `_inject_data_flip(circuit: MemoryCircuit, *, round_index: int, wire: int) -> str`
  - `_inject_measurement_flip(circuit: MemoryCircuit, *, round_index: int, ancilla_wire: int) -> str`
  - `_forced_signature(circuit: MemoryCircuit, source: str) -> tuple[tuple[int, ...], tuple[int, ...]]` — returns `(detector_indices, observable_indices)` in ascending order.

Both injectors take the circuit rather than the bare source string, so each can reject an out-of-range round, a wire the code does not declare, and a source it cannot anchor on, before returning anything.

Injection anchors, both pinned by observation of `flagquantum/qec/circuit.py:205-217`:

- Data flip: inserted immediately after the single occurrence of `"    for round_index in range(rounds):\n"`, guarded by `if round_index == <round_index>:`, emitting `qp.X(wires=<wire>)`. The guard fires the error at the **start** of that round, before any of its CNOTs — the convention settled in Design Decision 1.
- Measurement flip: inserted immediately **before** that check's `"        last = qp.measure(wires=<ancilla>)\n"` line, guarded the same way. Each check has a unique ancilla wire, so the anchor occurs exactly once; if it does not, fail closed rather than injecting into the wrong check.

`_forced_signature` lowers with `lower_dynamic_program(program, (circuit.rounds,), max_dynamic_measurements=circuit.rounds * len(circuit.code.checks))` and executes with `execute_hybrid_dynamic_session(lowered.circuit, shots=2, seed=0, strategy="trajectory")`. `strategy="trajectory"` is explicit and load-bearing: with `strategy="auto"` and `shots >= 32` the executor silently takes the batched path, which is a different code path.

**Both shots must agree.** If `classical_bits[0] != classical_bits[1]` or `samples[0] != samples[1]`, raise `ValueError` stating that the mechanism is not deterministic and therefore not a Pauli mechanism in the reference gate set. This is the fail-closed guard from Design Decision 4.

Detector bits come from the layout, never from the raw register: for each `Detector` in `circuit.detectors.detectors`, XOR the referenced measurements, where a `MeasurementRef` with `round_index is None` reads `samples[shot][wire]` and otherwise reads `classical_bits[shot][round_index * len(checks) + position_in_checks]`. **`position_in_checks` is the check's index in the `code.checks` tuple — never `CodeCheck.index`.** Observable bits come from `ObservableLayout`, XORing the terminal `samples` over `pauli.support`. A measurement flip in the final round consequently flips **two** detectors — its own round's and that check's terminal boundary — which is why a signature is a set of indices rather than a single one.

- [ ] **Step 1: Write the failing test**

Create `tests/qec/test_dem_signatures.py`:

```python
"""Coverage for the forced-error signature engine."""

from __future__ import annotations

import dataclasses

import pytest

from flagquantum.qec.circuit import MemoryCircuit, build_memory_circuit
from flagquantum.qec.codes import RepetitionCode
from flagquantum.qec.dem import (
    _forced_signature,
    _inject_data_flip,
    _inject_measurement_flip,
)

pytestmark = pytest.mark.integration


def _built(distance: int) -> MemoryCircuit:
    return build_memory_circuit(RepetitionCode(distance=distance), rounds=distance)


def test_data_flip_at_round_zero_wire_zero() -> None:
    built = _built(3)
    source = _inject_data_flip(built, round_index=0, wire=0)
    detectors, observables = _forced_signature(built, source)
    assert detectors == (0,)
    assert observables == (0,)


def test_data_flip_on_the_middle_wire_touches_two_checks() -> None:
    built = _built(3)
    source = _inject_data_flip(built, round_index=0, wire=1)
    detectors, _ = _forced_signature(built, source)
    assert detectors == (0, 1)


def test_data_flip_in_the_last_round_touches_only_that_round() -> None:
    built = _built(3)
    source = _inject_data_flip(built, round_index=2, wire=0)
    detectors, observables = _forced_signature(built, source)
    assert detectors == (4,)
    assert observables == (0,)


def test_measurement_flip_flips_its_round_and_the_next_round() -> None:
    """A flipped syndrome bit is compared against both of its neighbours."""

    built = _built(3)
    source = _inject_measurement_flip(built, round_index=0, ancilla_wire=3)
    detectors, observables = _forced_signature(built, source)
    assert detectors == (0, 2)
    assert observables == ()


def test_measurement_flip_in_the_last_round_reaches_the_terminal_boundary() -> None:
    built = _built(3)
    source = _inject_measurement_flip(built, round_index=2, ancilla_wire=3)
    detectors, observables = _forced_signature(built, source)
    assert detectors == (4, 6)
    assert observables == ()


def test_measurement_flip_on_the_last_check() -> None:
    built = _built(3)
    source = _inject_measurement_flip(built, round_index=0, ancilla_wire=4)
    detectors, _ = _forced_signature(built, source)
    assert detectors == (1, 3)


def test_scaled_distance_produces_the_expected_detector_count() -> None:
    built = _built(5)
    source = _inject_data_flip(built, round_index=0, wire=0)
    detectors, observables = _forced_signature(built, source)
    assert len(built.detectors.detectors) == 24
    assert detectors == (0,)
    assert observables == (0,)


def test_data_injection_anchor_must_be_unique() -> None:
    built = _built(3)
    mangled = dataclasses.replace(
        built, source=built.source.replace("    for round_index in range(rounds):\n", "")
    )
    with pytest.raises(ValueError, match="round loop"):
        _inject_data_flip(mangled, round_index=0, wire=0)


def test_measurement_injection_anchor_must_be_unique() -> None:
    built = _built(3)
    mangled = dataclasses.replace(
        built, source=built.source.replace("        last = qp.measure(wires=3)\n", "")
    )
    with pytest.raises(ValueError, match="measurement"):
        _inject_measurement_flip(mangled, round_index=0, ancilla_wire=3)


def test_injection_rejects_a_round_outside_the_configured_range() -> None:
    """A mechanism outside the configured rounds is rejected, not silently ignored."""

    built = _built(3)
    with pytest.raises(ValueError, match="round"):
        _inject_data_flip(built, round_index=3, wire=0)


def test_injection_rejects_an_undeclared_wire() -> None:
    built = _built(3)
    with pytest.raises(ValueError, match="declared"):
        _inject_data_flip(built, round_index=0, wire=99)
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
python -m pytest tests/qec/test_dem_signatures.py -q
```

Expected: `ImportError: cannot import name '_forced_signature'`.

- [ ] **Step 3: Implement the engine**

Implement in `flagquantum/qec/dem.py`. Imports to add at module top:

```python
from ..compiler._hybrid import INDEX, capture_source, lower_dynamic_program
from ..runtime.dynamic.hybrid_session import execute_hybrid_dynamic_session
from .circuit import MeasurementRef, MemoryCircuit
```

Helpers to implement, matching the Interfaces block above. Three details that are easy to get wrong:

1. The injectors validate before they edit: `round_index` must be in `range(circuit.rounds)`, `wire` must be in `circuit.code.data_wires` for a data flip and `ancilla_wire` in `circuit.code.ancilla_wires` for a measurement flip, and only then does the anchor search run. The messages must carry the words the tests match on: "round", "declared", and "round loop".
2. The data-flip anchor check must distinguish "not found" from "found more than once" and raise `ValueError` mentioning the round loop in both cases.
3. `_forced_signature` computes each detector's parity by XORing its referenced measurements and keeps the indices whose parity came out one, ascending. `Detector` already forbids repeating a measurement reference, so the XOR is over distinct references; the reason a signature can still hold two indices is that one measurement flip can move two different detectors.

- [ ] **Step 4: Run the test and confirm it passes**

```bash
python -m pytest tests/qec/test_dem_signatures.py -q
```

Expected: all tests pass. This file is marked `integration` because it lowers and executes; it takes a few seconds at `d = 5`.

- [ ] **Step 5: Commit**

```bash
git add flagquantum/qec/dem.py tests/qec/test_dem_signatures.py
git commit -m "feat: read a mechanism signature from one forced error"
```

---

### Task 7: Build a model from a memory circuit

**Files:**
- Modify: `flagquantum/qec/dem.py`
- Test: `tests/qec/test_dem_from_memory_circuit.py`

**Interfaces:**
- Consumes: everything from Tasks 1, 2, and 6.
- Produces: `DetectorErrorModel.from_memory_circuit(circuit: MemoryCircuit, *, noise: PhenomenologicalNoise) -> DetectorErrorModel` (classmethod), plus two module-private helpers — `_mechanisms(circuit: MemoryCircuit, noise: PhenomenologicalNoise) -> tuple[tuple[float, str], ...]` returning ordered `(probability, injected_source)` pairs, and `DetectorErrorModel._merge_mechanisms(entries, *, num_detectors: int, num_observables: int) -> DetectorErrorModel` where each entry is `(probability: float, detectors: tuple[int, ...], observables: tuple[int, ...])`.

Mechanism enumeration, in this order: for each round `r` in `range(rounds)` and each `wire` in `circuit.code.data_wires`, a data flip with probability `noise.data_flip`; then for each round `r` and each `check` in `circuit.code.checks`, a measurement flip with probability `noise.measurement_flip`. When a probability is zero the mechanism still has to be enumerated only if it can matter — it cannot, so skip zero-probability mechanisms entirely and let the caller's counts stay honest. Both loops iterate **`code.checks` in tuple order**, and the classical-bit offset uses that position.

Merging: mechanisms with identical `(detectors, observables)` combine by `p1 * (1 - p2) + p2 * (1 - p1)`. Mechanisms with an empty signature are discarded and counted; the count is not part of the public model.

- [ ] **Step 1: Write the failing test**

Create `tests/qec/test_dem_from_memory_circuit.py`:

```python
"""Coverage for building a detector error model from a memory circuit."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from flagquantum.qec.circuit import build_memory_circuit
from flagquantum.qec.codes import CodeCheck, RepetitionCode
from flagquantum.qec.dem import DetectorErrorModel
from flagquantum.qec.noise import PhenomenologicalNoise
from flagquantum.qec.pauli import Pauli

pytestmark = pytest.mark.integration


def test_repetition_model_shape() -> None:
    built = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    model = DetectorErrorModel.from_memory_circuit(
        built, noise=PhenomenologicalNoise(data_flip=0.05, measurement_flip=0.05)
    )
    assert model.num_detectors == 8
    assert model.num_observables == 1
    # 9 data mechanisms plus 6 measurement mechanisms, all with distinct signatures
    assert model.num_errors == 15


def test_every_mechanism_has_a_unique_signature_at_distance_three() -> None:
    built = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    model = DetectorErrorModel.from_memory_circuit(
        built, noise=PhenomenologicalNoise(data_flip=0.05, measurement_flip=0.05)
    )
    signatures = [(error.detectors, error.observables) for error in model.errors]
    assert len(set(signatures)) == len(signatures)


def test_a_noiseless_model_has_no_mechanisms() -> None:
    built = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    model = DetectorErrorModel.from_memory_circuit(built, noise=PhenomenologicalNoise())
    assert model.num_errors == 0
    assert model.num_detectors == 8


def test_data_flips_alone_always_flip_the_observable() -> None:
    built = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    model = DetectorErrorModel.from_memory_circuit(
        built, noise=PhenomenologicalNoise(data_flip=0.05)
    )
    assert model.num_errors == 9
    assert all(error.observables == (0,) for error in model.errors)


def test_measurement_flips_alone_never_flip_the_observable() -> None:
    built = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    model = DetectorErrorModel.from_memory_circuit(
        built, noise=PhenomenologicalNoise(measurement_flip=0.05)
    )
    assert model.num_errors == 6
    assert all(error.observables == () for error in model.errors)


def test_probabilities_come_from_the_noise_record() -> None:
    built = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    model = DetectorErrorModel.from_memory_circuit(
        built, noise=PhenomenologicalNoise(data_flip=0.25)
    )
    assert {error.probability for error in model.errors} == {0.25}


def test_distance_five_mechanism_count() -> None:
    built = build_memory_circuit(RepetitionCode(distance=5), rounds=5)
    model = DetectorErrorModel.from_memory_circuit(
        built, noise=PhenomenologicalNoise(data_flip=0.05, measurement_flip=0.05)
    )
    assert model.num_detectors == 24
    assert model.num_errors == 45


@dataclass(frozen=True)
class _IndexSwappedCode:
    """Two checks whose declared indices disagree with their tuple positions.

    Stage 1 recorded that ``CodeCheck.index`` is not consumed by the library and
    that a code may declare indices disagreeing with position with no error. The
    classical-bit stride is positional, so a model that keyed on ``index`` would
    scramble this code's signatures.
    """

    distance: int = 3

    @property
    def num_data_qubits(self) -> int:
        return 3

    @property
    def num_ancilla_qubits(self) -> int:
        return 2

    @property
    def data_wires(self) -> tuple[int, ...]:
        return (0, 1, 2)

    @property
    def ancilla_wires(self) -> tuple[int, ...]:
        return (3, 4)

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        return (
            CodeCheck(
                index=1,
                stabilizer=Pauli(z_wires=(0, 1)),
                ancilla_wire=3,
                cnot_wires=((0, 3), (1, 3)),
            ),
            CodeCheck(
                index=0,
                stabilizer=Pauli(z_wires=(1, 2)),
                ancilla_wire=4,
                cnot_wires=((1, 4), (2, 4)),
            ),
        )

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (Pauli(z_wires=(0, 1, 2)),)


def test_signatures_key_on_check_position_not_on_declared_index() -> None:
    """A measurement flip is read at the check's *positional* classical bit.

    The classical register strides by tuple position, so a model keyed on
    ``CodeCheck.index`` would read the wrong bit and produce a different set.
    """

    built = build_memory_circuit(_IndexSwappedCode(), rounds=2)
    model = DetectorErrorModel.from_memory_circuit(
        built, noise=PhenomenologicalNoise(measurement_flip=0.05)
    )
    assert model.num_detectors == 6
    assert model.num_errors == 4
    signatures = {error.detectors for error in model.errors}
    assert signatures == {(0, 2), (1, 3), (2, 4), (3, 5)}


def test_merging_combines_identical_signatures() -> None:
    """Two mechanisms with one signature merge by XOR probability."""

    merged = DetectorErrorModel._merge_mechanisms(
        [(0.1, (0,), ()), (0.2, (0,), ())], num_detectors=1, num_observables=0
    )
    assert merged.num_errors == 1
    assert merged.errors[0].probability == pytest.approx(0.1 * 0.8 + 0.2 * 0.9)


def test_distinct_signatures_are_kept_apart() -> None:
    merged = DetectorErrorModel._merge_mechanisms(
        [(0.1, (0,), ()), (0.1, (1,), ())], num_detectors=2, num_observables=0
    )
    assert merged.num_errors == 2


def test_empty_signature_mechanisms_are_discarded() -> None:
    merged = DetectorErrorModel._merge_mechanisms(
        [(0.1, (0,), ()), (0.1, (), ())], num_detectors=1, num_observables=0
    )
    assert merged.num_errors == 1
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
python -m pytest tests/qec/test_dem_from_memory_circuit.py -q
```

Expected: `AttributeError: type object 'DetectorErrorModel' has no attribute 'from_memory_circuit'`.

- [ ] **Step 3: Implement construction**

Add to `flagquantum/qec/dem.py`:

- `_mechanisms(circuit, noise) -> tuple[tuple[float, str], ...]` yielding each `(probability, injected_source)` pair, skipping zero probabilities.
- `DetectorErrorModel._merge_mechanisms(entries, *, num_detectors, num_observables) -> DetectorErrorModel` where each entry is `(probability, detectors, observables)`; signature-keyed accumulation plus the XOR merge, discarding empty signatures.
- `DetectorErrorModel.from_memory_circuit` wiring them together: enumerate, force each signature, merge, and return. It must validate that `circuit` is a `MemoryCircuit` and `noise` is a `PhenomenologicalNoise`, and it must surface the injection engine's `ValueError` unchanged when the source does not have the expected shape, so a hand-built `MemoryCircuit` whose source does not match its layouts fails closed with a stated reason rather than producing a wrong model.

- [ ] **Step 4: Run the test and confirm it passes**

```bash
python -m pytest tests/qec/test_dem_from_memory_circuit.py -q
```

Expected: all tests pass. The `d = 5` case lowers and executes 45 mechanisms and takes roughly one second.

- [ ] **Step 5: Commit**

```bash
git add flagquantum/qec/dem.py tests/qec/test_dem_from_memory_circuit.py
git commit -m "feat: build a detector error model from a memory circuit"
```

---

### Task 8: Cross-check against the circuit simulator

**Files:**
- Test: `tests/qec/test_dem_cross_check.py`

**Interfaces:**
- Consumes: `_mechanisms`, `_inject_data_flip`, `_inject_measurement_flip`, `_forced_signature`, `DetectorErrorModel.from_memory_circuit`, `PhenomenologicalNoise`.
- Produces: test-local helpers only. The proposal states that detector-rate comparison is a test and an evidence step, **not public API**, so nothing new is published from this task.

Two checks, both in the ordinary `integration` lane per Design Decision 10:

1. **Exhaustive pairwise linearity** at `d = 3`. For every pair of mechanisms, inject both, execute once, and assert the measured detector and observable bits equal the XOR of the two single-mechanism signatures. `C(15, 2) = 105` executions, about 1.2 seconds, fully deterministic — no statistic and no flakiness. This is the strongest structural evidence in the stage: it tests the XOR-composition assumption the model's merge arithmetic rests on, against the real simulator.
2. **Sampled-rate agreement** at `d = 3` and `d = 5`. Per shot, draw an independent Bernoulli per mechanism, inject the fired set, execute, and read the detectors and observables off the layouts. Compare the empirical rates to `detector_rates()` and `observable_rates()` with a tolerance stated in standard errors. Budget `d = 3`: 1000 shots (~15 s); `d = 5`: 500 shots (~11 s). Fixed seed, so the test is deterministic in practice.

The tolerance is `4 * sqrt(r * (1 - r) / shots)` with a floor that keeps a zero-rate detector from producing a zero-width band. State it in the test as a named constant with the reasoning, and assert on every detector and every observable rather than a summary statistic, so a single wrong mechanism is located rather than averaged away.

The test file's module docstring must record what the comparison does and does not prove, in the terms Design Decision 11 sets out.

- [ ] **Step 1: Write the test**

Write the two checks described above. Include a helper that turns a fired-mechanism set into one injected source, reusing the Task 6 helpers so there is exactly one injection implementation in the repository.

- [ ] **Step 2: Run the test and confirm both checks pass**

```bash
python -m pytest tests/qec/test_dem_cross_check.py -q
```

Expected: pass, in roughly 30 seconds. Confirm the timing so the `pr-runtime` budget assumption in Design Decision 10 holds; if the file exceeds 60 seconds, reduce the `d = 5` budget first, because the pairwise sweep at `d = 3` is the check worth protecting.

- [ ] **Step 3: Prove the check can fail**

Break the model deliberately — for example, make `_merge_mechanisms` add probabilities instead of combining them by parity — confirm the sampled-rate check fails, then restore. A comparison that cannot fail is not evidence. Record the observed failure in the commit message.

- [ ] **Step 4: Commit**

```bash
git add tests/qec/test_dem_cross_check.py
git commit -m "test: compare detector error model rates against the simulator"
```

---

### Task 9: Publish, document the boundary, and pass the repository gates

**Files:**
- Modify: `flagquantum/qec/__init__.py`
- Modify: `flagquantum/qec/IMPLEMENTATION.md`
- Test: `tests/qec/test_dem_public_surface.py` (create)

**Interfaces:**
- Consumes: every public name from Tasks 1–7.
- Produces: `PhenomenologicalNoise`, `DemError`, `DemSample`, and `DetectorErrorModel` published from `flagquantum.qec`, and the boundary described in `IMPLEMENTATION.md`.

- [ ] **Step 1: Write the failing test**

Create `tests/qec/test_dem_public_surface.py`:

```python
"""The stage 2a names are reachable from the qec package root."""

from __future__ import annotations

import pytest

import flagquantum.qec as qec

pytestmark = pytest.mark.unit


def test_stage_two_names_are_published() -> None:
    for name in ("DemError", "DemSample", "DetectorErrorModel", "PhenomenologicalNoise"):
        assert name in qec.__all__
        assert hasattr(qec, name)


def test_stage_one_names_are_still_published() -> None:
    for name in ("Pauli", "CodeCheck", "RepetitionCode", "build_memory_circuit"):
        assert name in qec.__all__


def test_frozen_names_are_unchanged() -> None:
    for name in (
        "Decoder",
        "StreamingDecoder",
        "RepetitionLookupDecoder",
        "RepetitionStreamingLookupDecoder",
        "RepetitionTemporalDecoder",
        "run_repetition_memory_experiment",
        "run_repetition_memory_noise_sweep",
    ):
        assert name in qec.__all__
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
python -m pytest tests/qec/test_dem_public_surface.py -q
```

Expected: `AssertionError` on the first missing name.

- [ ] **Step 3: Publish the names and document the boundary**

Add the imports and `__all__` entries to `flagquantum/qec/__init__.py`, keeping the existing ordering style. Then extend the "Code-independent records" section of `flagquantum/qec/IMPLEMENTATION.md` with a detector-error-model subsection stating, at minimum:

- What a DEM is here: every independent physical error mechanism identified with the detectors and observables it flips, exact for Pauli noise in the reference gate set.
- That construction is exact and does not sample, and that each mechanism's signature comes from one forced execution with a two-shot determinism assertion.
- That the model is built on the memory circuit **without** in-circuit feedback, because the model describes the mapping a decoder inverts.
- That it is defined for Pauli noise only, and that non-Pauli channels fail closed rather than being approximated.
- The two noise-scope facts: phenomenological noise is the primary evidence, and the cross-check compares DEM-predicted rates against an injected-shot sampler; the comparison shares the injection helper with construction, so what it tests is XOR composition and merge arithmetic, not the underlying physics independently. The one independent check — parsing the emitted text with the real `stim` package — is developer-time, not a committed test.
- That mechanical limits mean the frozen profile's `compiled_lookup` feedback path is not modelled.
- The scope refusals the proposal requires: no threshold claim, no logical-suppression claim, no real-time or hardware-feedback claim, and every DEM-sampled rate labelled as DEM-sampled.
- That `README.md` is deliberately not updated here, because the Stage 5 documentation sweep owns it.

- [ ] **Step 4: Run the full QEC suite and every repository gate**

```bash
python -m pytest tests/qec -q
black --check flagquantum tests
ruff check flagquantum tests
mypy --strict --python-version 3.12 --ignore-missing-imports flagquantum
python tools/check_architecture.py
python tools/check_repository_language.py
python -m pytest -m "(smoke or unit or integration) and not qiskit and not triton" -q \
  --cov=flagquantum --cov-report=xml
python tools/check_coverage.py
```

Expected: `tests/qec` passes with the Stage 1 count plus the new tests; the gates are clean; `[global] min = 76` and `[packages.qec] min = 82` both hold. If `packages.qec` is close to its floor, add unit tests for the uncovered branches rather than lowering the floor.

- [ ] **Step 5: Commit**

```bash
git add flagquantum/qec/__init__.py flagquantum/qec/IMPLEMENTATION.md \
  tests/qec/test_dem_public_surface.py
git commit -m "docs: publish the detector error model and describe its boundary"
```

---

## Self-Review

**Spec coverage.** Every Stage 2 deliverable in the proposal maps to a task: `DetectorErrorModel` and `DemError` to Task 2, `from_memory_circuit` to Task 7, `detector_error_matrix()` and `observables_flips_matrix()` to Task 3, `dem_sampling()` to Task 4, `to_stim_text()` and `from_stim_text()` to Task 5. `PhenomenologicalNoise` is Task 1. The proposal's acceptance criteria are covered as follows: the detector-count identity is asserted in Task 7 at `d = 3` and `d = 5`; "every signature is reproduced by a single-forced-error execution with no mechanism missing or duplicated" is the mechanism-enumeration and uniqueness assertions in Task 7 plus the pairwise sweep in Task 8; "DEM-predicted detector rates agree with rates sampled from the simulator inside a stated confidence interval at `d = 3` and `d = 5`" is Task 8's second check; "non-Pauli channels fail closed" is the determinism assertion in Task 6 and is enforced for circuit-level noise in 2b; the majority-versus-parity convention is already documented by Stage 1 and is not touched here; "no document claims a threshold, logical suppression, real-time decoding, or hardware feedback" is the boundary text in Task 9.

**Deliberately out of scope for 2a**, owned elsewhere: the decoder, `decoding_graph.py`, `matching.py`, and the `pymatching` extra (Stage 3); `statistics.py`, `wilson_interval`, and the distance-scaling evidence artifact (Stage 4); the capability row, the executable example, the QEC `README.md` sweep, and any capability promotion (Stage 5); circuit-level noise and the Kraus-to-Pauli decomposition (2b, its own plan).

**Known debt this stage must not walk past**, from the proposal's "Stage 1 landed state and known debt": the DEM keys on check tuple position, never `CodeCheck.index` (Tasks 6 and 7, with a purpose-built index-swapped code); the layouts are validated for membership only and not for semantics, so the forced execution is the authority on the signature rather than the layout's intent (Task 6); only Z-type checks are representable, which the repetition code and the 2a test codes all satisfy; `Pauli.from_text`'s format is pinned only by round-trip tests and the DEM is its first library consumer, so Task 5's stim text uses its own parser rather than reaching for `Pauli.from_text`; and the observable-count validator still has no test, which Stage 1 recorded and left — do not fix it here.

**Every task ends in a commit and a green gate.** Task 1's duration is minutes; Tasks 6, 7, and 8 lower and execute and are marked `integration`, so they run in `pr-runtime` and in the `coverage` job, and the 2a budget for the whole cross-check file is about 30 seconds.
