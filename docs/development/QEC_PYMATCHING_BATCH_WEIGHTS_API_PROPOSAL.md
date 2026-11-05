# SDK batch matching weights: restricted API and implementation proposal

Status: API owner approved eight paths; implementation candidate, acceptance pending.
Proposed title: feat(qec): return SDK batch matching weights.
Primary domain: QEC optional SDK adapter; local single_device_fast_path.

## Confirmed gap and existing delivery

Pinned PyMatching 2.4.0 source 6f63b2b9474ba0fa7e511fe52bffdce858a06984 exposes
Matching.decode_batch(return_weights=True), returning predictions and SDK weights
from the same native call. FlagQuantum's dense method returns predictions only;
PR1090 adds canonical-padding packed predictions, currently a locally validated
candidate with actual-main acceptance pending. Neither entry exposes SDK batch
weights. Sparse full-record batching retains MatchingDecodeResult.weight, which
is a recomputed double-precision selected-edge sum. It is a different quantity;
changing that field or substituting it for SDK weights is expressly excluded.

Verified sources: existing quality/pymatching/matching.py lines348-459 and
flagquantum/qec/adapters/pymatching.py weight semantics and batch implementation;
existing tests/qec/test_pymatching_batch.py and decoders/batch.py. No new study or
SDK execution is required to establish this interface gap.

## Proposed protected method

    PyMatchingDecoder.decode_batch_with_weights(
        self,
        detection_events: torch.Tensor,
        *,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]

The new tuple is (predictions, sdk_weights). No new result class, root export,
registry or serialized schema is introduced. The name is scoped by the existing
SDK-specific decoder; documentation must call the second component SDK-reported
matching weights, not original physical probabilities or exact graph weights.

Unpacked input preserves the dense method's binary bool/integer CPU contract.
Packed input preserves PR1090's dense CPU uint8 exact byte-width, LSB-first and
canonical zero padding contract. The two packed flags are independently strict
bools, supporting all four input/output combinations. Prediction output is owned
CPU bool[B,L] when unpacked, owned CPU uint8[B,ceil(L/8)] when packed; unused packed
output bits must be zero. Weights are an independently owned CPU float64[B]
Tensor, finite and nonnegative in the existing admitted nonnegative graph domain.
Validate the original external dtype/device/shape before any conversion; malformed
buffers must fail atomically without fallback or partial result. The admitted
upstream weights buffer must already be float64, not coerced from malformed ints
or reduced-precision output. Check that contract against the pinned binding
source before implementation and preserve an explicit mismatch if found.

Invoke _matching.decode_batch exactly once with return_weights=True and both
packed flags explicitly stated. Do not compute weights from a second decode,
matching record or different graph. Empty input returns both exact empty shapes
without SDK invocation. Nonempty zero-observable input still reaches the SDK and
returns per-shot weights. No mutable SDK buffer escapes or aliases input/output.

Original decode_batch, decode_packed_batch, decode, complete result fields,
constructor refusals and dependency errors retain their behavior. Private input
validation/copying may be extracted for reuse by the existing and new paths only
when existing default calls, errors, preparation order and output checks remain
identical. Do not route old methods through new weight validation: that could
introduce a failure absent from their protected contracts. Shared helpers must
have these actual consumers; no new provider abstraction is justified.

## Eight-path limit and dependency order

1. flagquantum/qec/adapters/pymatching.py: exactly one public method; narrow
   private validation reuse if needed; no changes to existing result semantics.
2. tests/qec/test_pymatching_batch.py: real pinned SDK conformance and independent
   small-model oracle; retain current optional guard and integration marker.
3. tests/unit/test_qec_installed_profiles.py: SDK-free pure contract/malformed
   output/ownership tests, reachable without NumPy or optional SDK imports.
4. tools/qec_installed_smoke.py: weighted call checks for PyMatching/both wheel
   and sdist profiles, preserving every existing sample and packed check.
5. docs/development/QEC_PYMATCHING_BATCH_WEIGHTS_API_PROPOSAL.md: approved
   signature, migration-free compatibility analysis, shortest user code.
6. contracts/qec-batch-prediction-v1.json: separately approved additive extension;
   leave every prior approval and dense/packed contract unchanged.
7. docs/development/QEC_FUNCTIONAL_005_BATCH_CONTRACT.md: distinguish SDK weight
   output from full-record selected-edge sums, and the four bit layouts.
8. docs/roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md: feature inventory row,
   exact head and separate local/candidate/main acceptance status.

Cut the independent branch from actual main after rechecking whether PR1090 has
landed. Do not merge the pending packed branch into another feature branch. If
main still lacks decode_packed_batch, prepare the patch against that actual base
without adding the unaccepted packed-only public method again; record layout
validation overlaps explicitly for later integration. CI watching is currently
paused by the owner; implementation authority does not include silently resuming
PR1087 recovery or calling either pending PR accepted. New-PR publication and its
automatic checks must be explicit in the eventual implementation authorization.

## Candidate user code (execution evidence recorded separately)

```python
import torch
from flagquantum.qec import DemError, DetectorErrorModel, PyMatchingDecoder

model = DetectorErrorModel(3, 3, (
    DemError(0.125, (0,), (0,)),
    DemError(0.125, (1,), (1,)),
    DemError(0.125, (2,), (2,)),
))
decoder = PyMatchingDecoder.from_detector_error_model(model)
shots = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.uint8)
predictions, sdk_weights = decoder.decode_batch_with_weights(shots)
print(predictions.tolist())
print(tuple(sdk_weights.shape), sdk_weights.dtype)
print(bool(torch.isfinite(sdk_weights).all()), bool((sdk_weights >= 0).all()))
```

Expected:

    [[False, False, False], [True, False, False], [True, True, True]]
    (3,) torch.float64
    True True

CPU only, existing optional PyMatching extra; Stim is not needed. Use an exact
candidate install for final proof, not a floating published package lacking this
method. No install or example execution is part of this proposal.

## Independent references and acceptance

The example graph has one unique boundary mechanism per detector, so predictions
are the syndrome and exact original graph weights are popcount * log(7). Check
SDK-reported weights against that independent oracle with the existing
num_edges * 2**-23 * max(1, weight) precision comparison bound documented by the
adapter, without weakening any old threshold or treating SDK weights as the
recomputed doubles. Also require exact equality with pinned direct SDK output
on the same graph/input. The empty syndrome's weight is exactly zero.

Include zero and positive weights, a disconnected/unreachable syndrome, equal
cost ties (compare paired values from one call, not cross-decoder prediction
identity), swapped logical columns, all eight small syndromes, byte crossings,
all four layout combinations, tail batches, noncontiguous and NumPy aliasing
cases. Do not promise seeded random-stream identity or general decoder equality.

Pure tests must include malformed tuple arity, prediction/weight shape and dtype,
wrong device, nonfinite/negative weights, nonzero padding, no SDK on invalid or
empty input, one SDK call on success, retained error cause, all-or-error failure
and unchanged old interfaces. Do not add skips or ignore rules. Keep Python3.12
explicit for the current NumPy typing environment; required supported-version
qualification remains a separate CI gate.

Proposed local commands after implementation:

```sh
python -m pytest -q tests/unit/test_qec_installed_profiles.py tests/qec/test_pymatching_batch.py tests/qec/test_adapters_pymatching.py tests/qec/test_independent_batch_decoding.py
python tools/check_dependency_policy.py
python -m ruff check flagquantum/qec/adapters/pymatching.py tests/qec/test_pymatching_batch.py tests/unit/test_qec_installed_profiles.py tools/qec_installed_smoke.py
python -m black --check flagquantum/qec/adapters/pymatching.py tests/qec/test_pymatching_batch.py tests/unit/test_qec_installed_profiles.py tools/qec_installed_smoke.py
python -m mypy --strict --python-version 3.12 --explicit-package-bases --follow-imports=silent flagquantum/qec/adapters/pymatching.py tests/qec/test_pymatching_batch.py tests/unit/test_qec_installed_profiles.py tools/qec_installed_smoke.py
```

Use existing environments and preserve all prior logs. Relevant owner,
architecture, English, hygiene, source-of-truth, docs-link and diff checks remain
required. PR description must carry the final-head actual user-code output and
precise environment. Fresh installed profiles and actual-main evidence, not
source-only tests, are required for a delivery receipt. No dependency/runner
change, research probe/worker/matrix, package publication or performance claim.

This advances the FP-06 return_weights row only. It does not close correlated
matching, all graph construction, broader packed input domains, full upstream
parity, native replacement, FTS02 scientific measurements or M2.
