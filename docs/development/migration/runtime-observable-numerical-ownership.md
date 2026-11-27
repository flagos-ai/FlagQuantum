# Observable readout numerical ownership

This independent family follows PR #1338 source
`d1d1d13b0cabab04b64250d57cd2772457eadb26`. Its formal activation follows the
parent's exact-head combined CI qualification. This child has separate full
source, installed artifact and published-head requirements; none certifies
#1084 integration.

Five numerical helpers use four existing Simulation files; no package,
framework dependency, public API or native ABI is introduced.

| Runtime caller | Simulation owner | Runtime responsibility retained |
| --- | --- | --- |
| JAX Statevector/TensorNetwork result `probabilities` / `expectation_z` | `numerics/probabilities.py` | Optional Torch admission, invalid-qubit validation, state reconstruction/cache, public aliases and result identity |
| `_ising_plan.execute_ising_plan` | `dynamics/ising_braid.py` | Plan/source admission, operator device placement, output naming and result/evidence |
| `_rydberg_presence.execute_presence_plan` | `dynamics/rydberg_presence.py` | Replay, seeded trajectory lifecycle, operator placement, time axis and result/evidence |
| Statevector `_local_expectation_z_hamiltonian_and_weights` | `statevector/adjoint.py` | Weight-cache key/admission, bounded memory/LRU, allocation, ownership and rollback |

## Exact numerical boundary

Raw probabilities stay `abs(state) ** 2`, without adding normalization. Dense Z
values keep the original index/sign construction, selected-qubit order and batch
shape. The Statevector plan's stored amplitude count and TensorNetwork's
computed count retain their existing callers. Missing Torch and invalid-qubit
refusals precede reconstruction or numerical dispatch as before.

The two physical output helpers retain their distinct original Einstein index
literals (`bij,ji->b` and `tij,ji->t`) and `.real` operation. Runtime still copies
the validated observable to its execution device before contraction. No new
configurable contraction abstraction is created.

Weighted readout moves the existing native attempt and its exact Tensor fallback
together. The same public native provider, environment flag, dtype/layout checks
and fallback expression apply. Runtime still owns and reuses the same weights;
the cache budget, LRU, isolation, disable switch and clearing behavior do not
move into Simulation. Existing independent validation/reference formulas and
Runtime statistics remain unchanged.

Seven donor computations map to the five helpers after explicit input-binding
normalization. All other existing function definitions and all product
assertions remain unchanged. Every new dispatch is exercised through its actual
Simulation owner by replacement tests, including provider acceptance/decline.

## Public boundaries retained and internal alias retired

The existing JAX result types, method signatures/annotations, `probability`,
`wavefunction` and TensorNetwork `to_statevector` aliases, reconstruction methods,
field order and historical type locators remain. These are existing result
interfaces and serialization identities under the
[public API protection policy](../PUBLIC_API_PROTECTION.md), not new
compatibility entries. The 102-entry facade inventory has no addition.

| Retired internal binding | Canonical owner and basis |
| --- | --- |
| `runtime.executors.statevector.reverse_adjoint.fused_observable_expectation` | The imported function remains publicly available at `simulation.native_cpu.fused_observable_expectation`, with its actual definition in `simulation.native_cpu.adjoint`. The executor directory explicitly [is not a public backend API](../../../flagquantum/runtime/executors/AGENTS.md); frozen contracts/fixtures contain no obligation for this Runtime alias. Its sole production use moves into the Simulation helper. Existing native tests import the canonical provider. No forwarding is retained. |

The `_require_torch` dependency remains an actual Runtime admission call; its
original refusal order is not replaced by importing Torch from Simulation first.
Runtime resolves the new weighted helper through its defining module, so an
implementation replacement does not require changing its consumer.

## Incoming source

The complete non-truncated #1084 Git tree at
`f820620faf9f15d3c95a700c55a6adbf06ffafbc` binds 2,222 product blobs. Six
existing candidate files are individually blob-verified. A later six-commit,
ninety-one-file increment to `667ccbaf69dbb572627e34177409b15627c3192f`
is also hash-verified, changes no product path and preserves all candidate
bindings. The subsequent one-commit, four-file increment to
`831e551dacf50c3abfae97131091757e21162b29` has likewise been blob-verified
and changes no product path; all twenty-one proposed paths are disjoint from
both increments. The latest changed incoming Python source set has 3,274 files.
All four public JAX
methods, both physical output expressions and the complete cached numerical
donor match the current parent. Three destination files were created by the
qualified parent series and are absent in the older incoming layout. Carry
those owners, earlier function splits and the new extractions together; do not
resurrect the old inline computations or Runtime native-function alias.

The 563-row current-parent ledger still produces seventeen raw old imports in
the changed incoming source set, without filtering or diagnostic normalization.
Those counts are not a complete branch/G0 acceptance. Retain the prior pending
Construction/IR/compiler, native/development distribution and root-tool/config
closures, including the complete CPU-nightly workflow/test path patch. Rust
remains deferred until the agreed closeout.

## Evidence

See [the acceptance record](runtime-observable-numerical-acceptance.md). A plain
source preview with no native binary must not be reported as equivalent to a
native installed artifact by dropping availability or provider-path checks.
