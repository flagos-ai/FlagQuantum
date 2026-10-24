# QEC-FUNCTIONAL-004 post-merge mainline acceptance

## Exact revisions

[#789](https://github.com/flagos-ai/FlagQuantum/pull/789) merged as
`3d253c9f9f899b3fb68396ccb4553d5df4754a24` on 2026-10-24, after explicit user
authorization to merge its latest green head. Reviewed head
`3540879e0f20e116bae8c0ff65bced6f50770011` passed
[complete PR CI, 54 jobs](https://github.com/flagos-ai/FlagQuantum/actions/runs/40065167670)
and [single-/required two-GPU validation](https://github.com/flagos-ai/FlagQuantum/actions/runs/40065167605).
Three optional manual pool checks were unselected, not newly skipped to pass.

This independent acceptance uses main
`4be2375eff47c112ce8d8e24f8206ed5592130d8`, including separately merged work after
#789. Its source, fresh installation artifacts and CI are checked independently
of the prior PR wheel. The documentation PR recording acceptance changes no
implementation, fixture, API contract snapshot, capability tier or release.

## Mainline CI

The complete mainline [CI run 40070940716](https://github.com/flagos-ai/FlagQuantum/actions/runs/40070940716) passed all 54 jobs at `4be2375eff47c112ce8d8e24f8206ed5592130d8`. The [GPU run 40070943609](https://github.com/flagos-ai/FlagQuantum/actions/runs/40070943609) passed source validation and the single-GPU and required two-GPU lanes; three pre-existing optional manual-pool jobs were unselected. Both the [immediate post-merge smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/40066803646) and [current-main smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/40067721978) passed. No checks or thresholds were changed.

The recorded dispatches name the exact revision above. A later moving main or
an older green PR run does not replace these conclusions. Development-container
publication is separate from QEC functional acceptance; this work creates no
release tag and publishes no package.

## Source and fresh installation

The complete source QEC suite, retirement API contract and both Simulation
fault-signature suites pass: **2,051 passed, one existing inapplicable-fault
skip**. This baseline includes the independently merged fixed-shot statistics
from #788; its additional tests are not attributed to #789.

A fresh isolated sdist and wheel-from-sdist contain all **814 package Python
members** byte-identical to this main revision, including all **53 QEC Python
members** and the Simulation primitive. SHA-256:

| Artifact | SHA-256 |
| --- | --- |
| `flagquantum-0.4.0.tar.gz` | `7281b2a422aa2b56240e76e01f1daad73131befc478f57e184cd20cbb4678ba3` |
| `flagquantum-0.4.0-cp310-abi3-macosx_11_0_arm64.whl` | `82b84ce73abfc2f265e48fce9c830031d97820c35fe0ee71f5286518a9176a2e` |

All **12 installed profiles** pass: Python 3.10, 3.11 and 3.12, each with
core only, Stim, PyMatching and both extras. Tests run outside the checkout with
`PYTHONPATH` unset and verify installed site-packages, dependency consistency,
the native extension, a Bell-state smoke, retirement and all six compatibility
suites. Each profile passes its 46 or 47 applicable compatibility cases.

The profiles complete **276 QEC numerical test executions**, **132 Simulation
numerical-owner test executions**, and **69 exact-output example executions**
(33 retained workflows, 12 existing circuit-noise examples, 12 correlated-noise
examples and 12 executions of the proposed batch contract's current-user baseline).
Three additional installed profiles with both SDKs execute the current #788
example and validate fixed-shot counts, logical failures, interval bounds and
sampling-source invariants. Random numerical counts are not frozen across SDK
versions. No required physical cross-check is replaced by a missing-extra skip.

The retirement gate preserves the historical ordered 108 root exports and the
two separately approved sampler additions from #784, class/pickle identities,
single registered `RotatedSurfaceCode` and registry policy. All original 15
restructuring fixtures remain unchanged. No compatibility snapshot is regenerated.

## Runnable correlated-noise example

With an installed distribution, or `PYTHONPATH=.` from the checkout:

```bash
python examples/qec/correlated_circuit_noise.py
```

Expected stdout:

```text
Detectors, observables: 6 1
Independent circuit locations: 2
Exclusive groups: 2
Sample shapes: (4, 6) (4, 1)
Independent-only text export: refused
```

The example builds exclusive two-qubit Pauli circuit faults and samples their
declared detector model. Physical circuit execution separately uses the existing
Stim extra. Independent-only text/graph/decoder consumers retain their refusals;
this is no correlated-decoder, threshold or performance claim.

## Next decision and concurrent work

[QEC-FUNCTIONAL-005's contract](QEC_FUNCTIONAL_005_BATCH_CONTRACT.md) specifies
Decision B against current main. It awaits explicit owner approval. The existing
native PyMatching Tensor batch method is already implemented and must not be
repeated or silently substituted for complete single-shot result records.

#788 already supplies fixed-shot logical counts and Wilson intervals in the
example, and #784 supplies compiled sampling. Those partial deliveries do not
activate the broader public experiment contract or complete generic batch
support. The independent AI assessment PR #786 is a separate workstream; this
record neither duplicates nor merges it. DynamicCircuit #731 remains untouched.
Only #789 received automatic merge authorization in this task; new PRs remain
unmerged unless separately authorized. No implementation of Decision B/C is
included in this acceptance/contract PR.
