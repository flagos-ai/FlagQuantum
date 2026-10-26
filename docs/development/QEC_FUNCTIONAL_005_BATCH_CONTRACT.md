# QEC-FUNCTIONAL-005: Independent-shot batch decoding contract

## Approval and implementation scope

This document makes Decision B of
[Proposal 087](API_CHANGE_PROPOSAL_087_QEC_SCALABLE_EXPERIMENTS.md#decision-b-additive-batch-operation)
concrete against main `4be2375eff47c112ce8d8e24f8206ed5592130d8`.
**The API owner explicitly approved this complete contract on 2026-10-26
(Asia/Shanghai)** in the delivery chat: approve Decision B / 005 and validate
and deliver according to the contract. This authorizes the operation below;
Decision C remains design-only and must not be implemented. The accompanying
reference implementation reuses each decoder's existing single-shot operation
and owning input validator. Current-head validation, normal merge and exact-main
acceptance remain separate gates.
Core owns decoder behavior; Integration reviews the additive public operation.
Target: unreleased development version 0.4.0, subject to release policy.

QEC-FUNCTIONAL-004 merged as
[#789](https://github.com/flagos-ai/FlagQuantum/pull/789). Its exclusive Pauli
models do not make an independent decoder into a correlated decoder. Preserve
all existing model and decoder refusals.

## Reuse and scope reconciliation

Main already supplies the separately approved
[PyMatching Tensor batch method](API_CHANGE_PROPOSAL_087_QEC_BATCH_DECODING.md),
[compiled memory sampling](QEC_REUSABLE_SAMPLING_PROPOSAL.md), and
[#788](https://github.com/flagos-ai/FlagQuantum/pull/788)'s fixed-shot counts and
Wilson intervals in the example. Do not reimplement them or claim that they
approve the generic batch or public experiment contracts.

| Surface | Carrier | Result and present status |
| --- | --- | --- |
| `PyMatchingDecoder.decode_batch` | CPU binary Tensor `[B, D]` | CPU bool logical predictions `[B, L]`; approved and implemented on main. |
| Approved `qec.decoders.batch.decode_batch` | Iterable of sparse per-shot detector-index iterables | Tuple of complete existing single-shot result records; Decision B approved on 2026-10-26; implementation accompanies this record. |
| Proposed public memory experiment | Separate configuration/result contract | Decision C remains unapproved. Existing example statistics are not this public API. |

The native PyMatching batch method returns no selected correction edges or their
weights. A wrapper must not manufacture those fields, drop them or call the
native method merely because its name matches the proposed operation.

## Approved public operation

The sole new canonical import is:

```python
from flagquantum.qec.decoders.batch import decode_batch
```

Approved common signature:

```python
def decode_batch(
    decoder: DetectorErrorModelDecoder,
    detection_events: Iterable[Iterable[int]],
) -> tuple[DetectorErrorModelDecodeResult, ...]:
    ...
```

No root or parent-package export is appended, no result type or registry is
created, and no decoder method becomes required. Type overloads may retain the
specific result types in the table below without changing runtime behavior.
There are no new options, backend selectors, async tasks or seed parameters.

Initial support is the exact in-tree concrete types below. Unknown subclasses
and plugins are not implicitly certified stateless by an inherited name or a
structural `decode` member. A non-decoder carrier raises `TypeError`; an
otherwise valid decoder outside the supported types raises `CapabilityError`,
including for an empty batch. Preserve single-shot plugin registration and
execution unchanged; a future opt-in contract requires a separate review.

| Concrete decoder | Complete per-shot result preserved |
| --- | --- |
| `MinimumWeightMatchingDecoder` | `MatchingDecodeResult`: observables, selected edges and weight. |
| `SlidingWindowMatchingDecoder` | `MatchingDecodeResult`, retaining configured windows, commit size and defect limits. |
| `BeliefPropagationDecoder` | `BeliefPropagationDecodeResult`, including mechanisms, weight, convergence and iterations. |
| `BeliefPropagationOsdDecoder` | `BeliefPropagationOsdDecodeResult`, including mechanisms, weight, convergence and iterations. |
| `PyMatchingDecoder` | Its existing `MatchingDecodeResult` from single-shot decoding, including actual edges and weight. Optional SDK requirements are unchanged. |

`SlidingWindowDecoder` is a stateful stream even though it also has a whole-block
`decode` method. Refuse it without resetting, consuming or mixing its buffered
rounds. Repetition history decoders, neural predecoders and arbitrary plugins
are not implicitly detector-model independent-shot implementations.

## Input, ordering and failure behavior

- The outer input is a finite iterable of shots. Each shot uses that decoder's
  existing sparse detector-index semantics. It is not a dense bit row. Reject
  Tensor carriers with actionable guidance to the existing adapter-only dense
  method; do not guess a representation from row length or binary values.
- Consume the outer iterator and every row iterator once. Retain one immutable
  tuple per row, without mutating caller-owned inputs. Validate the complete
  materialized batch before any decoding where validation is already knowable.
  Eager storage is proportional to input indices plus output records; this is
  not a streaming or bounded-memory claim for an unbounded generator.
- Preserve existing integer, boolean, bounds and duplicate semantics per
  decoder. In particular, `BeliefPropagationDecoder` treats repeated indices as
  one set member, while matching and BPOSD reject duplicates. Do not globally
  deduplicate rows, reinterpret duplicates as parity cancellation or impose a
  new duplicate policy on accepted single-shot inputs. Reuse the owning
  validation helpers instead of inventing a second syndrome grammar.
- A supported empty batch returns `()`. A nonempty batch returns one complete
  record per row, in input order. Results match repeated calls to that
  decoder's existing `decode` under its existing numerical and tie conventions.
- After static validation, run algorithmic decoding in row order. Existing
  capability, invalid-syndrome, defect-budget, non-convergence and dependency
  errors propagate. No failed row becomes a zero correction or disappears from
  the denominator. No decoder, model or backend is silently substituted.
- Delivery is all-or-error: no partial result tuple escapes on failure. This
  does not promise transactional rollback of private SDK caches. The supported
  operations do not carry a stream of committed rounds across shots.

Initial implementation uses existing single-shot operations and makes no speed
claim. It must not call `PyMatchingDecoder.decode_batch` and misrepresent its
logical-only output as full correction records. The already implemented native
logical-only operation remains the supported fast path for callers needing that
specific result. Any later optimization under QEC-FUNCTIONAL-006 must preserve
all promised fields or obtain a separate contract approval; it must not add a
second native graph/cache or repeat an already shipped batch method.

## Compatibility and independent acceptance

Preserve the ordered historical 108 root exports and the two separately
approved sampler additions, all existing signatures/defaults, class/pickle
identities, single registered rotated class, registry replacement rules and
all original frozen fixtures. No mandatory dependency or checkpoint schema is
added. No change to or merge of DynamicCircuit #731 is included.

The implementation PR must provide separately named conformance tests covering:

1. Two independently implemented algorithms, then every supported type, with
   complete record equality against direct per-shot decoding. Check edges and
   weights, mechanism numbering, convergence and iterations, not only logical
   bits. Include tie/precision cases under each implementation's own convention.
2. Empty/single/multiple shots, generators with observable consumption counts,
   ordering and input immutability; wrong carriers, invalid integers, out-of-range
   indices and the differing duplicate policies. Static-invalid input must cause
   no decode calls. Test an invalid late row and an algorithmically refused row
   after valid rows; neither yields a partial successful batch.
3. Graphlike and hyperedge domains, unreachable syndromes, finite defect/window
   budgets, convergence/refusal cases, optional dependencies and exclusivity
   refusals. The new function must not strip ids from #789's models.
4. A plugin that implements only the old single-shot protocol must still
   register and decode unchanged. Unknown classes/subclasses and a partially
   consumed streaming decoder must be refused without state mutation.
5. Core and optional-SDK installation profiles, source and wheel/sdist, runnable
   public examples and full required CI including single-/two-GPU checks at the
   reviewed head. Do not weaken thresholds or update frozen snapshots to pass.

## Runnable current-user baseline

The existing single-shot workflow below remains runnable with core Torch only,
using an installed distribution or
`PYTHONPATH=.` from a source checkout:

```python
from flagquantum.qec import DetectorErrorModel, get_decoder

model = DetectorErrorModel.from_stim_text(
    "error(0.125) D0 L0\ndetector D0\nlogical_observable L0\n"
)
shots = ((), (0,))
for name in (
    "minimum_weight_matching", "belief_propagation", "belief_propagation_osd"
):
    decoder = get_decoder(name, model)
    results = tuple(decoder.decode(shot) for shot in shots)
    print(name, tuple(result.observables for result in results))
```

Expected stdout:

```text
minimum_weight_matching ((), (0,))
belief_propagation ((), (0,))
belief_propagation_osd ((), (0,))
```

The implementation must add a runnable example for the actual approved import
and preserve these results. This proposal neither implements the new function
nor relabels the existing native Tensor method as a full-record operation.

## Implementation and verification

The sole new module is `flagquantum.qec.decoders.batch`. It uses exact concrete
type admission, snapshots finite outer/row iterables once, rejects Tensor
carriers explicitly, validates the complete input through the existing owning
helpers, and calls the original `decode` in row order. It returns the untouched
complete records, not reconstructed wrappers. No root/parent export, result
schema, decoder registration, numerical kernel, new dependency or speed claim
is added. Existing factory/decode capability refusals and algorithm options stay
with their owners.

The new conformance suite exercises core algorithms without optional SDKs and
adds the real PyMatching type when installed. Absence is tested as the existing
dependency refusal, not a replacement decoder. Core and SDK counts are reported
separately; five-type qualification requires the actual SDK profile. Cases pin
complete records and independently stated logical answers, once-only generators,
late invalid input with zero decode calls, input isolation, differing duplicates,
empty batches, unknown subclasses/plugins, an already consumed stateful stream,
algorithmic failure after a valid row, ties/near ties, hyperedges, finite budgets
and nonconvergence. The existing exclusive-model constructor refusal is retained;
no error ids are stripped. A trap prevents use of the logical-only PyMatching
Tensor method for these complete records.

The new runnable [five-decoder example](../../examples/qec/independent_batch_decoding.py)
requires `python -m pip install -e '.[pymatching]'`, CPU only. Stim is not needed
for this example's existing owned DEM text reader. Its complete code also appears
in the implementation PR, with exact observed output. Each line ends in
`((), (0,), (0,))`; the five names are minimum_weight_matching,
sliding_window_matching, belief_propagation, belief_propagation_osd and pymatching.
The operation is registered in the existing public-docstring execution list;
the module-coverage assertion is unchanged. Local focused validation passed
83 conformance cases with the real SDK and 68 in core, plus the public doctest
runner. Twelve source/wheel/sdist installation profiles passed across core,
Stim-only, PyMatching-only and combined environments. These local native builds
used the existing Torch 2.13 runtime and do not certify the Torch 2.10 build floor.
The bounded local default run was incomplete. Its new doctest-registration
failure was corrected; four native-fusion expectation failures reproduced on
the exact unmodified main baseline `b88bffb1cc45910800f50e9312c80162fde1aea3`
with matching worker/thread settings. Local default is not reported green.
Actual source/install/wheel/sdist and current-head CI/GPU acceptance is recorded
in that PR's delivery record. No future merge SHA is inserted recursively into
this source document.
## Delivery acceptance

The reference implementation merged through
[#904](https://github.com/flagos-ai/FlagQuantum/pull/904) as
`92f5269caa126cd4220e7612a5d2207c644e6fd3`, from reviewed head
`3ff5fca2de51b18ee1f2cb88e7fbd5fa96dee029`. Complete head CI `40280575633`
succeeded in all 54 jobs; GPU `40280575629` succeeded with three successful jobs
and three original optional skips. All 60 checks succeeded or retained those
skips. Exact-merge Main smoke `40283856972` succeeded; it is not complete main CI.
Development-container publishing `40283856979` was cancelled, with all four
matrix jobs cancelled and no established cause. It was not retriggered and is
not reported successful. The [central progress ledger](../roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md)
retains these boundaries and the remaining functional work. Decision C stays
design-only; this completed batch contract does not authorize that implementation.
