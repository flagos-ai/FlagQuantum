# API Change Proposal 035: QPU digital-Twin model persistence

## Status

**Approved and implemented as a compatible Twin v1 addition.** On 2026-09-14,
the API owner authorized the next Twin increment after resumable submissions.

## Problem

FlagQuantum can persist Twin evidence, repeated validation, and an in-flight
submission, but not the calibrated Twin model that creates predictions. After a
process restart, users must otherwise retrieve calibration again or reconstruct
the exact `NoiseModel` and physical mapping by hand. A newer calibration would
create a different Twin identity and could not reproduce the frozen prediction.

## Public API

Save an already constructed provider-neutral or Quafu-native Twin:

```python
import flagquantum as fq
from flagquantum.remote.qpu import QuafuProvider

provider = QuafuProvider()
chip_info = provider.fetch_chip_info("Shenglian")
twin = fq.twin.from_quafu_chip_info(
    chip_info,
    target="quafu:Shenglian",
    qubits=(20, 27),
)
fq.twin.dump_twin(twin, "shenglian-twin.json")
```

Restore it in another process and predict without provider access:

```python
import flagquantum as fq

restored = fq.twin.load_twin("shenglian-twin.json")
circuit = fq.Circuit(2).h(0).cx(0, 1)
prediction = restored.predict(circuit)

print(restored.snapshot.identity)
print(prediction.twin_probabilities)
```

Both functions are offline. They do not fetch calibration, submit hardware,
poll a task, train or update a model, publish evidence, or make routing choices.

## Artifact and safety

`flagquantum.qpu_digital_twin.v1` stores the immutable snapshot together with
the complete provider-neutral `NoiseModel`: device calibration, gate-duration
data, readout rules, and Kraus channels. Loading reconstructs those framework
objects and verifies both calibration and noise-model identities against the
snapshot before returning a usable `QPUDigitalTwin`.

The loader rejects unknown top-level or snapshot fields, unsupported nested
schemas, non-canonical noise models, malformed JSON, non-finite values, and
identity mismatches. The writer creates a private mode-0600 file, allows an
idempotent write of the identical Twin, and refuses to replace different or
invalid content.

The artifact contains no provider credential or task receipt. It is a
reproducible model artifact, not hardware-validation evidence, a signature, a
training checkpoint, or proof that the model remains accurate after its
captured calibration time.

## Compatibility

This proposal adds `dump_twin()` and `load_twin()` without changing a frozen
Twin v1 name, signature, field, status, or serialized meaning. The new artifact
schema composes the existing frozen `TwinSnapshot` and Noise v1 schemas.
