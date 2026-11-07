# Matched detector pairs: restricted API and delivery proposal

Status: API owner approved; implementation candidate in consolidated PR1087.
Proposed title: feat(qec): expose matched detector pairs.
Primary domain: QEC optional decoder adapter, single_device_fast_path.

## Demonstrated gap and reuse

Pinned PyMatching 2.4.0 source6f63b2b9474ba0fa7e511fe52bffdce858a06984,
matching.py lines526-622, exposes decode_to_matched_dets_array and
 decode_to_matched_dets_dict. These return endpoints of matched detection paths,
not all correction edges. Existing MatchingDecodeResult.error_edges states the
symmetric difference of selected correction chains. Reconstructing a unique
matching endpoint assignment from that edge set is not guaranteed, especially
with zero-cost/tied solutions. Current adapter has no public matched-event view.

Reuse the existing graph, _defects input validation, stored SDK matching graph,
optional dependency error and graph-translation refusals. No new graph model,
registry, compatibility wrapper, result class or dependency is justified.

## Protected API proposal

    PyMatchingDecoder.decode_to_matched_detectors(
        self, detection_events: Iterable[int]
    ) -> tuple[tuple[int, int | None], ...]

Input is the same sparse detector-index carrier as existing decode. Reuse its
normalization: integral indices excluding bool, in graph bounds, no duplicates;
materialize a generator exactly once. Dense binary Tensor input is not this
operation. Preserve current constructor and faithful-translation refusals.

Output is an immutable, deterministic presentation of the SDK's selected pairs:
ordinary pairs are (min_detector, max_detector), boundary matches are
(detector, None), and the pairs are sorted by first endpoint. Each active detector
appears exactly once, with no inactive endpoint. The virtual boundary may be used
by multiple detectors; it is not a detector and never gets a numeric public ID.
Empty defects return () after validation without an SDK call. All values are
FlagQuantum-owned Python ints/None; no NumPy array or SDK object escapes.

Call decode_to_matched_dets_array exactly once on the stored SDK graph, translating
its second-endpoint -1 marker to None. Validate a two-column signed-integer CPU
buffer, dimensionality, endpoint ranges, non-self pairs, no repeated/inactive
endpoints, complete coverage of active defects, and the boundary-marker position.
No float-to-int coercion, alternate decoder, second solution, partial result or
mutable borrowed buffer is allowed. SDK inability to match maps to the adapter's
existing CapabilityError pattern retaining the ValueError cause. Malformed SDK
buffers also raise CapabilityError. Original _defects TypeError/ValueError stay
unchanged. Canonical sorting changes presentation only, not the selected pairing.

This is a separate observation call, not an extension of MatchingDecodeResult.
Do not claim that its pairing, an independently called logical decode and a
weighted batch necessarily describe one identical tied solution. No added field,
changed record weight, root re-export, serialization promise or API alias.

The upstream array and dictionary views map to this one canonical domain result.
Conversion to a mate dictionary is user-side ordinary data manipulation. On
unique solutions both upstream representations are independent conformance
references. At ties, compare validity and cost/domain conditions rather than
requiring separate SDK calls to select identical mates.

## Seven-path implementation limit

1. flagquantum/qec/adapters/pymatching.py — one added public method, reuse existing
   graph and defect validation; old decode/batch methods remain unchanged.
2. tests/qec/test_adapters_pymatching.py — marked optional SDK differential cases,
   independent unique-solution references, ties and unreachable syndromes.
3. tests/unit/test_qec_installed_profiles.py — SDK-free fake-boundary validation,
   malformed results, ownership and call-count cases; no new skips.
4. tools/qec_installed_smoke.py — installed PyMatching/both wheel and sdist
   checks; preserve all old sample/batch cells already on actual main.
5. docs/development/QEC_PYMATCHING_MATCHED_DETECTORS_API_PROPOSAL.md — approved
   signature, output semantics, compatibility and shortest runnable code.
6. flagquantum/qec/adapters/README.md — local domain entry and shortest edit/test
   path explaining pairs versus correction edges and boundary identity.
7. docs/roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md — exact full-parity rows and
   separate implementation/local/actual-main statuses.

No new contract file or serialized schema is needed for an immutable view. The
approved API proposal and conformance tests are its source of truth; do not add
single-shot fields to the unrelated batch-prediction manifest.

## Planned user code, not executable on the current adapter

```python
from flagquantum.qec import DemError, DetectorErrorModel, PyMatchingDecoder

model = DetectorErrorModel(3, 1, (
    DemError(0.125, (0,), (0,)),
    DemError(0.1, (1, 2), ()),
))
decoder = PyMatchingDecoder.from_detector_error_model(model)
pairs = decoder.decode_to_matched_detectors((0, 1, 2))
print(pairs)
mates = {}
for first, second in pairs:
    mates[first] = second
    if second is not None:
        mates[second] = first
print(mates)
```

Expected:

    ((0, None), (1, 2))
    {0: None, 1: 2, 2: 1}

This has two disconnected graph components: detector0 has the unique boundary
fault and detectors1/2 have one joint fault. Therefore the matching is unique
by graph connectivity, independently of SDK output. Use installed candidate CPU
with the existing PyMatching extra; Stim is not required. Do not run this planned
new API until implemented and approved.

## Acceptance and independent references

- Unique two-endpoint path with intermediate inactive nodes proves that pairs are
  not correction-edge lists. The SDK pair endpoints omit internal path vertices.
- Multiple boundary matches, mixed ordinary/boundary paths, unsorted generators,
  empty input, one active defect, nonempty zero-logical graphs, and unreachable
  odd syndromes without a boundary.
- Direct pinned upstream array and mate-dictionary conversion on unique models;
  independently enumerate small matchings and shortest path costs. Reuse the
  existing weight-precision comparison bound without changing any old threshold.
- Equal-cost solutions need no universal mate identity; assert a valid matching
  over the exact active set, not the outcomes of unrelated calls or decoders.
- Pure cases reject wrong rank/dtype/device, malformed pair length, boundary in
  first position, negative/out-of-range/inactive endpoints, self pairs, duplicate
  coverage or missing active detectors. Mutating the returned SDK buffer later
  cannot change returned Python tuples. No SDK call on invalid/empty input; one
  call on success; failures retain their cause without returning partial tuples.
- Constructor missing-extra behavior and all existing record/batch contracts
  remain unchanged. Mandatory installed wheel profiles and sdist use the existing
  smoke machinery; no dependency installation is part of proposal preparation.

Proposed implementation validation:

```sh
python -m pytest -q tests/unit/test_qec_installed_profiles.py tests/qec/test_adapters_pymatching.py tests/qec/test_pymatching_batch.py
python tools/check_dependency_policy.py
python -m ruff check flagquantum/qec/adapters/pymatching.py tests/qec/test_adapters_pymatching.py tests/unit/test_qec_installed_profiles.py tools/qec_installed_smoke.py
python -m black --check flagquantum/qec/adapters/pymatching.py tests/qec/test_adapters_pymatching.py tests/unit/test_qec_installed_profiles.py tools/qec_installed_smoke.py
python -m mypy --strict --python-version 3.12 --explicit-package-bases --follow-imports=silent flagquantum/qec/adapters/pymatching.py tests/qec/test_adapters_pymatching.py tests/unit/test_qec_installed_profiles.py tools/qec_installed_smoke.py
```

Check the actual base test file's existing typing before using the full-file
strict command; no ignore or rule relaxation may be used to paper over a
pre-existing test-boundary issue. Preserve the first failed log and stop for
review under the owner's instruction. Use existing quality tools/environments.
Owner, architecture, language, hygiene, docs and diff checks remain required.

## Publication and overall progress

After explicit API/implementation approval, cut directly from actual main and
reuse any merged delivery. Never import pending feature branches to manufacture
main acceptance. PR1087, PR1090 and PR1091 remain separate candidates until their
actual-main receipts exist. Identify concrete path overlap against main before
implementation; maintain the separate history of each approved scope.

Publish only after local checks and final-head public example proof. The PR body
must contain real user code, prerequisites, command, observed output and exact
head. Do not wait for or recover CI this round under the current steering, and
do not waive future required gates or call the PR delivered before actual-main
acceptance. No research probe, worker, full matrix, package release, dependency
or runner change is authorized.

This advances FP-06 matched-detector views in the bounded admitted graph domain.
General graph/matrix construction, correlated matching, broader upstream input
domains, native replacement, performance parity and other Stim surfaces remain
open. No percentage is inferred from API or PR counts.
