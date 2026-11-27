# Runtime readout sampling numerical ownership

This batch follows PR #1337 source head
`f019aafc634640c008b421468c6a50e4c50a840b`. The formal batch activates only after that parent's combined exact-head CI
qualification. Its own source and installed qualification do not certify a
published child commit; that exact head requires separate CI.

Five existing numerical bodies move into three existing Simulation modules.
No new package, optional dependency, public API or numerical algorithm is added.

| Existing caller | Numerical owner | Boundary |
| --- | --- | --- |
| `runtime/measurements.py::_sample_pauli_product` | `simulation/numerics/probabilities.py` | Clamp/normalize the Pauli distribution, then draw/unpack bits. Runtime obtains the distribution and creates the generator after normalization. |
| `runtime/dynamic/execution.py::_sample_trajectory` | `simulation/statevector/measurement_ops.py` | Draw/unpack one final state; Runtime applies readout errors and Pauli-frame corrections. |
| `runtime/dynamic/execution.py::_run_dynamic_batched` | `simulation/statevector/measurement_ops.py` | Draw/unpack one result per trajectory row; Runtime keeps branch counts, readout errors and execution evidence. |
| `runtime/results/trajectories.py::MPSMonteCarloResult.sample` | `simulation/mps/sampling.py` | Draw from the retained empirical mixture without dense reconstruction; the published method retains argument/retention validation. |

## Numerical and failure invariants

The extracted statement ASTs are identical after replacing only explicit input
bindings and using qubit vocabulary for newly introduced private parameters.
Existing definitions in all three destination modules are unchanged.

Keep the following order and behavior, including refusal paths:

- Pauli sampling clamps and normalizes before Runtime creates its generator;
  a failing normalization must not move behind generator creation. The two
  numerical helpers preserve that ordering without a generator callback.
- Final Dynamic draws keep the existing `torch.multinomial` defaults. They do
  not adopt the different replacement policy of static Statevector sampling.
  Single and batched shapes, integer packing and generator consumption remain.
- Mixture sampling checks retained-state compatibility, selects each noise
  trajectory uniformly and calls the retained state's sampler in the original
  order. It retains the same per-batch placement and generator state.
- The wider-than-63-qubit integer-index refusal still occurs after the original
  draws. Moving it earlier would change its RNG side effect and is excluded.
- Runtime statistical summaries, resource admission, noise/feedback decisions,
  error counters and result schemas retain their existing responsibilities.

## Retained public boundaries and deleted implementation

`MPSMonteCarloResult` and its published `sample` signature, annotations,
validation and `flagquantum.runtime.trajectories.result` pickle locator remain.
This follows `PUBLIC_API_PROTECTION.md`, the frozen public class annotations
fixture and original trajectory/result compatibility tests. The method now
invokes its actual numerical owner; its former sampling loop is deleted.
No new compatibility inventory entry is needed.

The private Pauli and Dynamic caller functions remain because they still own
request resolution, RNG policy, readout noise, frame/feedback decisions or
result assembly. Their extracted numerical statements are removed rather than
copied. No old private member is re-exported as a forwarding alias. These
modules still have concrete Runtime duties and are not empty transition files.

## Incoming integration

The exact full #1084 files at
`f820620faf9f15d3c95a700c55a6adbf06ffafbc` were blob-verified, including the
original `runtime/trajectories/result.py` owner. The incoming Pauli function,
mixture method and two Dynamic donor blocks have identical ASTs. Incoming MPS
sampling definitions match the parent. Statevector's existing
`_expectation_pauli_string` difference is prior parent integration work and is
not modified by this batch; carry the qualified parent's changes forward.
The probability module is already created by the parent series.

Carry all five extractions when integrating that branch: its older result path
must not resurrect the deleted sampling loop. The complete qualified f019 parent
ledger (560 rows) reports seventeen raw old imports in the incoming source set: fifteen prior gated
obligations, one new CPU-nightly test reader of `tools.ci_tier` (canonical owner
`tools.ci.ci_tier`), and one already-covered instruction donor. The new test
reader, its workflow command and both test command lookups must be updated
together during joint integration without restoring forwarding.
This audit is not G0 acceptance, and
Construction/IR, native ABI, developer distribution and Rust remain gated.

## Verification

Read [the acceptance record](runtime-readout-sampling-acceptance.md) for raw
source, installed-artifact, regression and exact-head CI qualification status.
Original frozen contracts, scientific assertions, evidence files and prior
failure records are preserved.
