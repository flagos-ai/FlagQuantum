# QEC reusable memory sampling proposal

Status: proposed for API review, not approved or implemented. Work A of the
[delivery plan](../roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md).

## Existing implementation and reuse decision

`qec/execution/sampling.py::_sample_record` lowers the memory, derives noise
locations, constructs a noisy IR and calls `sample_noisy_measurements` once.
`simulation/stabilizer/engine.py::sample_noisy_measurements` builds a Stim circuit
and invokes `compile_sampler(seed=...)` on every call. The graph in the existing
PyMatching adapter is already reused and needs no new caching abstraction.

`runtime/dynamic/hybrid_session.py::execute_hybrid_dynamic_session` executes a
compiler-lowered fixed-round measurement-feedback profile. It is a function,
not a retained compiled-sampling resource. It validates special metadata and
bound gates/rotations before invoking dynamic execution; it does not expose
prepared Stim state or a batch sampler. Forcing memory sampling through it
would change that profile without solving sampler ownership.

Therefore retain current Runtime policy and introduce only the minimal prepared
sampling boundary. This is not a general Runtime session redesign.

## Candidate user API

These names and signatures are proposals, not existing exports:

```python
compile_memory_sampler(
    circuit: MemoryCircuit,
    *,
    noise: PhenomenologicalNoise | NoiseModel,
    seed: int | None = None,
) -> MemorySampler

MemorySampler.sample(*, shots: int) -> DemSample
MemorySampler.close() -> None
MemorySampler.__enter__() -> MemorySampler
MemorySampler.__exit__(exc_type, exc_value, traceback) -> None
```

`MemorySampler` has a private constructor: users obtain it from the factory.
The final typed context-manager signature must be specified in the approved
contract, not copied literally from this abbreviated notation. Proposed root
exports are exactly the factory and returned handle class; no SDK type enters
those signatures. The existing one-shot APIs and defaults remain unchanged.

Candidate workflow after implementation:

```python
with compile_memory_sampler(memory, noise=noise, seed=53) as sampler:
    first = sampler.sample(shots=1024)
    second = sampler.sample(shots=1024)
```

The handle returns detection/observable batches only in this slice. Raw
measurement sampling remains on the existing interface; do not add an output
mode or additional result type without a current consumer.

## Ownership and state

- QEC execution owns measurement planning, existing noise locations and parity
  extraction. It must not import Stim or implement its numerical kernel.
- Simulation owns the private prepared circuit/sampler and its release. The
  public QEC handle composes that seam and does not expose SDK handles.
- Validation, lowering, noise placement and compilation occur at construction,
  before a sampler is returned. No global hash cache or shared mutable sampler.
- Configuration is a snapshot: copy/resolve values affecting execution at
  compilation. Mutating the original circuit/noise afterwards does not retune
  the handle; construct a new handle. This avoids an implicit cache invalidation
  protocol. The snapshot must not retain a mutable configuration alias used by
  later parity extraction.
- A handle belongs to its creator process and thread. Cross-owner sampling or
  close fails explicitly. Separate workers construct separate handles; worker
  seed assignment remains orchestration policy.
- Sample calls advance the same RNG stream. A fixed seed and identical call
  sequence reproduce within the supported engine/platform scope. Different
  batching, engines or architectures need distribution checks, not bit equality.
- `close` releases references to prepared state, is idempotent for its owner,
  and makes later sample calls raise RuntimeError. Context exit closes on both
  normal and exceptional paths and never suppresses the user's exception.
- No pickling, checkpointing or serialization of the live handle. Resume at a
  new independently seeded work unit; do not claim restoration of an opaque RNG.

## Validation and errors

Reuse current circuit/noise/seed validation and optional-dependency errors at
construction. Unsupported capabilities fail before any sample draw. `shots`
accepts a positive integer with bool rejected, matching the one-shot sampling
contract; zero is not a second empty-batch policy. Invalid shot counts do not
consume RNG. Invalid lifecycle operations raise RuntimeError.

An execution failure closes/poisons the handle before propagating the original
error, since SDK state advancement cannot be safely rolled back. Further sample
calls fail. Explicitly unsupported serialization raises TypeError. No silent CPU
fallback, approximation or automatic SDK installation is introduced.

## Implementation split after approval

1. Review the two public names, signatures, snapshot policy, owner restrictions
   and error behavior; land the explicit additive contract and conformance data.
2. Simulation implements private preparation/release behind its existing seam.
3. QEC execution composes immutable plans and parity extraction; root exports
   only the approved names. Update the repetition workflow to reuse one handle.
4. Run ownership, exact-map, independent distribution, compatibility, installed
   artifact and bounded-memory evidence. Avoid broadening the noise grammar here.

## Acceptance scenarios

- Instrument preparation to assert exactly one lowering and SDK compilation
  across multiple sample calls, and no extra compilation on close.
- Recompute detector/observable parity from a controlled measurement record;
  compare original and prepared paths on deterministic noise fixtures and
  independent sampled distributions.
- Compare two independently prepared handles with identical seed and call
  sequence. Do not assert one large batch equals partitioned batches.
- Cover context success/failure, repeated close, sample-after-close, execution
  failure poisoning, process/thread misuse and serialization refusal.
- Verify invalid shots draw nothing and original caller mutation cannot alter a
  compiled experiment's mapping or noise.
- Verify missing Stim at construction with core import still working. Replace
  the private preparation seam for deterministic mapping conformance, and use a
  real independent reference for numerical evidence; a fake alone is not proof.
- At fixed batch size, increasing requested total shots does not grow retained
  sample buffers. Measure preparation and steady-state sampling separately.

Open review decision: approve this narrow compile/sample/close public API and
snapshot/ownership semantics, or revise it before implementation. This document
records no human approval and creates no runtime implementation.
