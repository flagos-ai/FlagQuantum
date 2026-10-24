# FTS-01 post-merge mainline acceptance

## Revisions and review gates

[PR #799](https://github.com/flagos-ai/FlagQuantum/pull/799) merged as
`e723477bb9b0398cca348addaa923b0e82322d4a` on 2026-10-24 after the user authorized
ongoing in-scope delivery and exact-head conditional merges. The reviewed head
`ebaee8faa5650a08654493999f1e7bcb3e955ffb` passed
[complete CI, all 54 jobs](https://github.com/flagos-ai/FlagQuantum/actions/runs/40077220900)
and [source/single-/required two-GPU checks](https://github.com/flagos-ai/FlagQuantum/actions/runs/40077220924).
Three pre-existing optional manual pool checks were unselected. GitHub reported
MERGEABLE/CLEAN, with no submitted reviews or unresolved review threads; the
merge was bound to the verified head rather than bypassing protection.

Main advanced during dispatch. This independent acceptance binds
`be4b9d2f347e67a4c3f2eeb095762bb40bb89d2e`, including separately merged #786,
#792, #795 and #797. Their implementations and additional tests are not
attributed to the documentation-only #799 change.

## Mainline CI and source verification

Main `be4b9d2f347e67a4c3f2eeb095762bb40bb89d2e` passed [complete CI, 54 jobs](https://github.com/flagos-ai/FlagQuantum/actions/runs/40079493702), [source/single-GPU/required two-GPU validation](https://github.com/flagos-ai/FlagQuantum/actions/runs/40079501527) and [smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/40079443850). Three pre-existing optional manual GPU-pool checks were unselected. The source QEC, retirement-contract and both Simulation fault-signature suites pass with **2,093 passed and one pre-existing inapplicable-case skip**. A fresh isolated sdist and wheel-from-sdist preserve all **818 package Python members** exactly; a clean Python 3.12 / Torch 2.14.1 core installation passes dependency consistency, native/Bell sampling and public examples outside the checkout. This is one fresh core installation, not a claim of a new local 12-profile run. Supported Python/SDK coverage is provided by exact-head CI. Latest proposal base `bfb1fd6f9d042a24515295e311daf15930af9cf2` adds only #800 documentation and has the identical package tree.

The separately merged #800 document proposes an annotated Stim bridge but does
not implement or approve its public entry points. The FTS-02A documentation
branch starts from its main merge `bfb1fd6f9d042a24515295e311daf15930af9cf2`.
No current-main result is inferred from an older green run; both exact revisions
and their package-tree equivalence are recorded.

## Fresh artifacts and installation

| Artifact | SHA-256 |
| --- | --- |
| `flagquantum-0.4.0.tar.gz` | `df1d06c2d5ea312bbbcce14a514cf609394eddfc6b4da685801ca994afcff261` |
| `flagquantum-0.4.0-cp310-abi3-macosx_11_0_arm64.whl` | `2a955d72c905db94618c47ceb6577b00dbc080638e7b9aace150a9fc5ebaf2bb` |

The sdist and wheel-from-sdist were built afresh from the exact main source.
Every one of the 818 package Python members matches its source bytes in both
archives. A new Python 3.12.14 environment installed the wheel with the default
resolver, selecting Torch 2.14.1. Dependency consistency, installed-site-packages
location, native import and Bell sampling pass with PYTHONPATH unset outside
the source checkout. Static Bell samples include their existing batch dimension
and have shape `(1, 32, 2)` with equal measured bits.

Both original master-plan public examples pass with exact expected stdout.
The [new H-T-H and feedback reference](FTQC_REFERENCE_WORKLOADS_AND_BUDGETS.md#runnable-current-api-example-non-clifford-interference-and-feedback)
also passes in this fresh installation. No package publication or release tag
was created. The earlier local twelve-profile matrix was not rerun here and
is not a certificate for newly changed modules; current Python/SDK validation
comes from the exact-head complete CI.

## Runnable user workflow

Save this as a Python file and run it with the reviewed installed wheel, or use
`PYTHONPATH=.` from the checkout. Core Torch is sufficient; no credentials or
hardware are needed.

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2).h(0).measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)
result = fq.run(circuit, options=fq.ExecutionOptions(shots=128, seed=53))
assert result.samples is not None
assert result.samples.shape == (128, 2)
print("Feedback parity:", bool((result.samples[:, 0] == result.samples[:, 1]).all()))
```

Expected stdout:

```text
Feedback parity: True
```

This certifies the local public workflow and a correlation invariant, not a
complete distribution test, encoded fault-tolerant feedback, domestic hardware
or experimental control.

## Baseline observations and next gate

Existing-producer sanity runs at the accepted implementation revision include:

- `dynamic_unified_run` with 128 batched shots, 8 trajectory shots, 128 static
  shots, one warmup, three timing repetitions and five planning observations.
  All of its already approved wrapper budgets pass. This short smoke is not
  the proposed comparative study or a new performance commitment.
- Surface X memory at d=3, three rounds, 128 shots, batch size 32: circuit
  sampling, complete counts and existing Wilson-interval invariants pass.
  Random failure counts are not pinned across SDK versions or hosts.
- The existing d=3, three-round circuit-location DEM producer completes in a
  clean CPU process and retains local non-release/capacity-refusal labels.
  Python traced allocation peaks and process RSS remain distinct quantities.

Raw outputs are retained with their existing producers and environment details;
no new benchmark registry or performance-release claim is introduced here.

[FTS-02A](FTQC_REFERENCE_WORKLOADS_AND_BUDGETS.md) specifies the next workload and
budget decisions. New numerical commitments, public interfaces, approximation,
package admission and hardware scope still require their explicit approvals.
Preserve the historical QEC exports plus approved additions, surviving class/
pickle identities, registration, one RotatedSurfaceCode and the original frozen
fixtures. No change to or merge of #731 is included.
