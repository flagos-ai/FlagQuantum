# Simulation direct-relay retirement

Prerequisite: qualified PR [#1351](https://github.com/flagos-ai/FlagQuantum/pull/1351),
head `c96e63362807b04adb083d656d8086aea50d8f3f`. The main-based carry commit
`ef988a853` preserves its exact tree. This batch removes seven internal argument
delegations and their historical Runtime image-helper binding. It introduces no
replacement forwarding or compatibility layer.

## Ownership and migration

| Removed internal definition | Existing numerical owner |
| --- | --- |
| `simulation.statevector.cx_adjoint.cpu_cx_permutation_images` | `simulation.native_cpu.permutation.compact_cx_permutation_images` |
| `simulation.tensor_network.entrypoints.build_tensor_network` | `simulation.tensor_network.local.build_local_tensor_network` |
| `simulation.tensor_network.entrypoints.build_tensor_network_expectation` | `simulation.tensor_network.observables.build_tensor_network_expectation` |
| `simulation.tensor_network.entrypoints.build_tensor_network_hamiltonian_expectation` | `simulation.tensor_network.observables.build_tensor_network_hamiltonian_expectation` |
| `simulation.tensor_network.entrypoints.build_tensor_network_hamiltonian_expectations` | `simulation.tensor_network.observables.build_tensor_network_hamiltonian_expectations` |
| `simulation.tensor_network.entrypoints.tensor_network_expectation_ps` | `simulation.tensor_network.observables.tensor_network_expectation_ps` |
| `simulation.tensor_network.state.tensor_network_expectation_ps` | `simulation.tensor_network.observables.tensor_network_expectation_ps` |

Paths above are relative to `flagquantum`. Runtime executors, existing numerical
tests, distributed scenario scripts, and the CUDA validation probe call those
owners directly. The probe import adjustment lands as a separate Integration
commit before Simulation implementation. `TensorNetworkState` methods retain
local observable imports after the original validation and admission stages.
The canonical numerical owner files are unchanged.

The image helper is removed from the old historical identity tuple. The ledger
no longer claims that its deleted implementation moved into `cx_adjoint`.
Four partial retirement records describe the seven deleted definitions without
claiming deletion of the still-used modules or relocation of new definitions.

## Protected and substantive boundaries

`entrypoints.tensor_network_expectations` remains protected by the stable
extension namespace in `contracts/public-api-v1-candidate.json` and its recorded
signature in `contracts/public-api-v0.2-baseline.json`. Its removal requires the
public API change process; this batch does not change that contract.

The existing NumPy MPS local-score kernel, JAX matrix composition and stacking
kernels, native Clifford availability boundary, embedded sampler constructor,
and host double-single encoding boundary retain their actual numerical,
backend, or conversion responsibilities. They are not argument-only relays.
Existing source-identity assignments for surviving CX numerical functions remain
intact. No new public compatibility is introduced.

## Verification and limits

The retained qualified native-install corpus contains 63 original outcomes,
including 43 returns and 20 refusals. The modified source reproduces the complete
encoded tensor and gradient bytes, plan/cache graphs, input changes, and refusal
results exactly. This includes the original empty-batch refusals; it does not
claim new batching or capacity support.

Every changed Python file passes a full non-import AST comparison against the
qualified parent under the two canonical binding renames and seven deletions.
The obsolete test that only asserts delegation through the deleted wrapper is
removed. All remaining test assertions, workloads, tolerances, markers, validation,
and scientific limits remain unchanged. Strict typing passes for all 1,264 package
Python files. The complete real CPU JAX selection passes 207 tests with six
existing local-device conditions, both from source and its own native wheel.
The three affected integration files pass 259 tests with 128 existing source
conditions and 386 tests with one optional-cotengra condition from the own native
installation. All 63 complete original outcomes match in both environments;
nine caller/defining modules retain the qualified predecessor's optional-heavy
cold-import sets, and all retired definitions and the old Runtime binding are
absent. All 1,264 product Python files match source, wheel, and installation;
both native extension binaries match wheel and installation.

The initial integration attempt retained two stale test consumers and three
sandbox-blocked loopback Gloo failures. Their raw logs are retained. The stale
consumer was updated, the obsolete delegation-only test removed, and the same
complete selection passes with local Gloo communication permitted. No product
calculation, test tolerance, backend, or refusal condition was changed.
PR #1352 head `9a3f580d7979115412adb686ed9fea6f8ec450aa` is now qualified:
all three workflows pass on attempt 1, with 432 successful checks and four
existing conditional skips. All 384 CPU reports are retained and verified;
Runtime accounts for 14,259 surviving nodes and coverage for 38,563, with the
obsolete wrapper test as the sole removed node. All 83 original scientific
singletons and call phases remain. Maximum whole executed partition times are
Runtime 485 seconds, coverage 586, unit 464, and distributed CPU 425, including
preparation/upload and excluding queue.

The complete default run contains the same 22,785 nodes: 21,920 pass and 865
conditionally skip. One evidence-provenance case skips after a historical Git
lazy-fetch timeout; the unchanged exact test subsequently passes with implicit
fetching disabled, against the real object store and declared unavailable
origins. Those separately retained results reconcile to the predecessor's
21,921 passed and 864 original conditions. The full-run skip is not rewritten.

PR #1084 head `5fa323b1356b91996afe5cae3c5bf043f65c15cd` remains draft and
conflicting. Its two new commits after `179ebbf6` change three independently
hash-verified documentation/test blobs and no production Python. They do not
intersect this batch. The earlier complete static review of its 2,101 production
Python files remains applicable to these candidates; this is not integrated G0
acceptance or completion of the overall architecture migration.

## Existing user workflow

```python
import flagquantum as fq

result = fq.run(fq.Circuit(2).h(0).cx(0, 1))
print(result.state.abs().square())
# tensor([[0.5000, 0.0000, 0.0000, 0.5000]])
```

Users keep the same API. Internal consumers use the defining numerical modules.
This batch does not change runner routing, resource quotas, scientific workloads,
or execution semantics, and makes no accelerator or scalability claim.
