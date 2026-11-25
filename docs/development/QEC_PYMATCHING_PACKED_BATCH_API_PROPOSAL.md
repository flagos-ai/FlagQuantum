# Proposed first parity gap: packed PyMatching logical batches

Status: approved by the API owner; implementation candidate, acceptance pending.
Proposed title: feat(qec): support packed PyMatching batch predictions.
Proposed branch: feat/qec-pymatching-packed-batches, cut directly from main.
Implementation base: main 2a5a81bbf841b7e88086f215bd07c533143813c3. The owner
subsequently directed independent development without waiting for PR1087 CI.
This branch does not incorporate PR1087; its unaccepted delivery remains separate.

## Demonstrated gap and reuse

Pinned PyMatching 2.4.0 Matching.decode_batch exposes bit_packed_shots and
bit_packed_predictions. The existing PyMatchingDecoder.decode_batch accepts only
a dense binary CPU Tensor with one detector per column and has no such options.
The existing sparse complete-record batch is a separate operation and cannot
stand in for packed logical predictions. Existing type/refusal tests are retained.

Reuse the current constructor, graph translation, optional dependency error,
SDK logical prediction graph and batch shape/error validation patterns. Add no
registry, backend manager, dependency or new graph representation.

## Protected API proposal

Add exactly one method on the existing PyMatchingDecoder:

    decode_packed_batch(self, detection_events: torch.Tensor) -> torch.Tensor

Input: dense CPU uint8 Tensor, shape (shots, ceil(num_detectors / 8)). Detector
index d is bit (d % 8) of byte (d // 8), least-significant bit first. The initial
slice requires canonical zero unused detector padding bits; this is an explicitly
narrower input domain to be tracked against upstream, not a full-method parity
claim. Output: newly owned dense CPU uint8 Tensor with shape
(shots, ceil(num_observables / 8)); unused logical padding bits are zero.

Use SDK decode_batch with both packed flags explicitly true. Reject incompatible
input before SDK invocation; preserve primary SDK error propagation/wrapping
consistent with the existing logical-only operation. Do not invoke a dense
fallback or modify caller input. Empty batches return exact packed shape without
SDK invocation. Zero observables, non-contiguous views, width crossing 8 bits,
malformed SDK outputs, poisoning/ownership and missing-extra behavior need tests.
No new resource limits or arbitrary size thresholds are introduced.

The current dense method, full correction records, precise record weights,
constructor refusals and all existing stable signatures remain unchanged.
No root export is added. This method explicitly belongs to the PyMatching SDK
adapter; it does not claim native decoding. A future native packed boundary needs
its own consumer-unchanged replacement test and explicit API proposal.

## Eight-path limit

1. flagquantum/qec/adapters/pymatching.py — one added method/private validation reuse.
2. tests/qec/test_pymatching_batch.py — existing marked test file; pure ownership,
   padding/shape/failure cases and optional SDK differential cases, no new skips.
3. docs/development/QEC_PYMATCHING_PACKED_BATCH_API_PROPOSAL.md — approved API
   scope, compatibility analysis, behavior and shortest runnable user example.
4. contracts/qec-batch-prediction-v1.json — additive approved method description;
   preserve every original signature/default/result/refusal assertion. Add a
   separately dated approval entry; do not broaden the historical approval scope.
5. docs/development/QEC_FUNCTIONAL_005_BATCH_CONTRACT.md — distinguish sparse
   full records, dense predictions and packed predictions; no claim all are one API.
6. docs/roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md — exact pinned parity rows,
   candidate/accepted states and remaining upstream options.

7. tests/unit/test_qec_installed_profiles.py — SDK-free packed validation tests
   using controlled fake SDK buffers, plus existing available-profile assertions.
   The existing integration file has a module-level importorskip, so pure cases
   must not be placed there. Preserve that guard and original test selection.
8. tools/qec_installed_smoke.py — mandatory packed example in PyMatching/both
   wheel and sdist profiles, leaving all original sample/report cells intact.

## Proposed user code and expected output

The candidate implements this public workflow:

```python
import torch
from flagquantum.qec import DemError, DetectorErrorModel, PyMatchingDecoder

model = DetectorErrorModel(3, 3, (
    DemError(0.1, (0,), (0,)),
    DemError(0.2, (1,), (1,)),
    DemError(0.3, (2,), (2,)),
))
decoder = PyMatchingDecoder.from_detector_error_model(model)
packed = torch.tensor([[0], [1], [2], [4], [7]], dtype=torch.uint8)
print(decoder.decode_packed_batch(packed).tolist())
```

Expected: [[0], [1], [2], [4], [7]]. Each detector has one unique boundary fault,
so logical prediction equals its detector bit. This exact combinatorial oracle
is independent of the SDK. Extend to more than eight detector/logical bits,
swapped logical columns and all eight three-detector syndromes to detect ordering
or padding bugs. Compare the same buffers with pinned upstream packed output and
existing dense predictions; do not require different decoding algorithms to agree
on arbitrary graphs or ties. Input/output alias tests must include NumPy views.

## Proposed acceptance commands and conditions

CPU and an existing environment with Torch and optional PyMatching are sufficient;
Stim is not required for this declared-model workflow. No installation is part of
this review. Future implementation acceptance commands:

```sh
python -m pytest -q tests/unit/test_qec_installed_profiles.py tests/qec/test_pymatching_batch.py tests/qec/test_adapters_pymatching.py tests/qec/test_independent_batch_decoding.py
python tools/check_dependency_policy.py
python -m ruff check flagquantum/qec/adapters/pymatching.py tests/qec/test_pymatching_batch.py tests/unit/test_qec_installed_profiles.py tools/qec_installed_smoke.py
python -m black --check flagquantum/qec/adapters/pymatching.py tests/qec/test_pymatching_batch.py tests/unit/test_qec_installed_profiles.py tools/qec_installed_smoke.py
python -m mypy --strict --explicit-package-bases --follow-imports=silent flagquantum/qec/adapters/pymatching.py tests/qec/test_pymatching_batch.py tests/unit/test_qec_installed_profiles.py tools/qec_installed_smoke.py
```

Use the repository-required pinned quality versions and supported Python matrix;
verify optional-SDK marker selection before pushing. Core-only pure tests must
remain reachable without SDK installation. After local success, user approval
must cover publication and normal final-head CI/GPU plus actual-main acceptance.
Reuse the existing package acceptance machinery and run the example against the
actual candidate install through the explicitly included smoke-tool path. Preserve raw failures;
no automatic rerun, dependency/runner change or threshold weakening.

No scientific probes, sampling workers, Linux admission, timing study or full
matrix is required or authorized. This slice closes one packed logical prediction
workflow; return_weights, correlations, broad graph construction, noncanonical
padding compatibility and the rest of the complete parity inventory stay open.


## Local candidate verification

The initial focused SDK suite passed 151 tests with three original source-profile
selection skips. A subsequently added zero-observable SDK case passed separately.
All 15 packed pure cases passed in the existing core-only environment without
Stim, PyMatching or NumPy. Strict Mypy 2.0 with explicit Python 3.12, Ruff 0.15
and Black 24.1 passed for the four changed Python files. The explicit Python
version matches the installed NumPy stubs; it is not certification of all Python
versions or a change to repository typing rules. Initial quality failures remain
in external delivery logs. Four installed wheel profiles and sdist-both use the
updated smoke hook; fresh candidate installation acceptance remains pending.


## Consolidated delivery

The owner directed consolidation into PR1087. These approved scopes are now
reviewed together; separate historical PRs remain preserved until replacement
is verified. Candidate integration does not imply actual-main acceptance.
