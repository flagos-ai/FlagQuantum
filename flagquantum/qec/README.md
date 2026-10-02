# Quantum error correction

Build experiments that connect syndrome extraction, decoding, correction, and
logical-result analysis. The long-term direction is a complete QEC workflow
for fault-tolerant quantum computing research, including logical operations
and hardware feedback.

QEC owns codes, decoder semantics, detection events, and Pauli frames. It
composes Compiler control flow, Runtime feedback, Simulation kernels, Noise
models, and Remote hardware interfaces.

## Start with a memory experiment

This experimental local example uses the three-data-qubit repetition-code
profile and injects one X error before syndrome extraction:

```python
from flagquantum.qec import ErrorEvent, ErrorSchedule, run_repetition_memory_experiment

result = run_repetition_memory_experiment(
    error_schedule=ErrorSchedule((ErrorEvent(round_index=0, wire=1),)),
    rounds=3,
    shots=16,
    seed=0,
)
print(result.logical_error_rate)
```

## Declare a code and its detection layout

`codes.py` holds the code records and `circuit.py` turns one into a memory
experiment. This is the code-independent layer beside the frozen repetition
profile:

```python
from flagquantum.qec import RotatedSurfaceCode, build_memory_circuit

memory = build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=3)
print(len(memory.detectors), len(memory.observables))
```

A detector is a measurement parity that is deterministic in the noiseless
circuit. Because both the initial state and the terminal data readout are in the
Z basis, a Z-type check declares a detector in every round plus one terminal
detector, while an X-type check declares one for every round after the first. A
code whose declared logical observable is not Z-type is refused rather than
measured under premises that do not hold for it.

## Decode a detection-event syndrome

`decoding_graph.py` turns a detector error model into a weighted graph and
`matching.py` decodes a syndrome on it. This is the code-independent decoder: it
reads the model's mechanisms and nothing about the code that produced them.

```python
from flagquantum.qec import (
    DetectorErrorModel,
    MinimumWeightMatchingDecoder,
    PhenomenologicalNoise,
    RepetitionCode,
    build_memory_circuit,
)

model = DetectorErrorModel.from_memory_circuit(
    build_memory_circuit(RepetitionCode(distance=3), rounds=3),
    noise=PhenomenologicalNoise(data_flip=0.02, measurement_flip=0.02),
)
decoder = MinimumWeightMatchingDecoder.from_detector_error_model(model)
sample = model.dem_sampling(shots=1, seed=0)
syndrome = tuple(int(i) for i in sample.detectors[0].nonzero().flatten())
print(decoder.decode(syndrome).observables)
```

An edge weight is `log((1 - p) / p)`, a mechanism that flips one detector becomes
an edge to a boundary node, and the decoder's prediction is the exclusive-or of
the selected mechanisms' observable labels. A mechanism that flips three or more
detectors is a hyperedge and is refused with a stated reason rather than
projected onto a pair. The matcher enumerates the ways to pair the defective
detectors, so it refuses a syndrome larger than its defect budget instead of
returning a pairing that only looks cheapest.

One fault stated twice is a second refusal, and merging is the repair. A model
whose mechanisms repeat a signature — the same detectors *and* the same
observables — is one a matcher must not weight, because the graph would hold two
parallel edges and take the cheaper, so the decoder refuses it and names the
merge that resolves it:

```python
from flagquantum.qec import DetectorErrorModel

split = DetectorErrorModel.from_stim_text(
    "error(0.01) D0 L0\n"
    "error(0.01) D0 L0\n"
    "detector D0\n"
    "logical_observable L0\n"
)
print(split.mechanisms_are_unique())  # False
whole = split.merge_duplicate_mechanisms()  # DemMergeRule.INDEPENDENT_PARITY
print(whole.num_errors, whole.errors[0].probability)  # 1 0.0198
```

`merge_duplicate_mechanisms(rule=...)` gives every shared signature one prior.
`DemMergeRule.INDEPENDENT_PARITY`, the default, is the probability that an odd
number of the group fires; `DemMergeRule.CLAMPED_LINEAR_SUM` adds them and clamps
at one, for a caller whose mechanisms exclude each other. The parity rule is
exact rather than tidy: a detector's rate is a product of `1 - 2p` factors over
the mechanisms that touch it, so the combined prior of a group is the single `p`
whose factor is that group's product and every rate is unchanged.
`mechanisms_are_unique()` and `require_unique_mechanisms()` are the predicate and
the refusal.

## Build a model from matrices, without a circuit

A code whose checks and logical operators are known as matrices needs no gadget
and no wire layout. `code_matrices` reads a code record into the Z-type
check matrix and the Z-type logical matrix, and `from_code_matrices` derives the
mechanisms from them:

```python
from flagquantum.qec import (
    DetectorErrorModel,
    PhenomenologicalNoise,
    RepetitionCode,
    code_matrices,
)

hz, lz = code_matrices(RepetitionCode(distance=3))
model = DetectorErrorModel.from_code_matrices(
    hz=hz,
    lz=lz,
    noise=PhenomenologicalNoise(data_flip=0.02, measurement_flip=0.02),
    num_rounds=3,
)
print(model.num_detectors, model.num_errors)  # 6 15
```

`hz[k, q]` is one when a bit flip on data qubit `q` flips check `k`, and `lz`
uses the same convention for the logical operators; `lz` may be omitted, and then
no mechanism flips an observable. A matrix that is not binary, is not
two-dimensional, or disagrees with `lz` about the number of data qubits is
refused rather than coerced, because a real-valued matrix is a rate description
and rounding it would decide which checks a fault triggers.

This route is the code-capacity experiment, so its detector geometry is not the
memory circuit's. A fault in round `r` reaches the detector band of round `r` and
the band of round `r + 1`, and the final round has no band after it, so the
detector count is `num_rounds * num_checks` with no terminal readout — where a
memory circuit gains a terminal detector per Z-type check because it measures its
data qubits. Both geometries are pinned to their own route and neither stands for
the other. The same limit as the circuit route applies to the fault family: the
data flip is a bit flip, so an X-type check and an X-type logical operator have
no row here, and the phase-flip family is not expressible.

## Sample the circuit itself, not the model

`sampling.py` samples the same experiment from the circuit rather than from the
model, so a logical-failure rate no longer depends on the statevector ceiling that
bounds construction. It shares the code record and the noise record with the
model above and nothing else: the model derives each mechanism's signature from
the source, this module derives each mechanism's instruction position from the
lowered program.

```python
from flagquantum.qec import (
    PhenomenologicalNoise,
    RepetitionCode,
    build_memory_circuit,
    sample_memory_circuit,
)

memory = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
sample = sample_memory_circuit(
    memory,
    noise=PhenomenologicalNoise(data_flip=0.02, measurement_flip=0.02),
    shots=1000,
    seed=0,
)
print(sample.detectors.shape, sample.observables.shape)
```

A data flip is placed at the round boundary *before* the round's first gate, so it
opens the frame that round's detectors compare against, and a measurement flip is
placed immediately *before* the readout of the check it corrupts. Neither position
exists in the source program, whose bounded hybrid capture refuses a channel call
outright, so both are derived from the lowered program — and the program must
lower to `rounds` identical blocks measuring each check once in the code's
declared order. A program that lowers to anything else, a channel that is not the
bit-flip pair, and a lowered measurement node are each refused with a stated
reason rather than sampled under an attribution that may be wrong.

Because the two samplers are separate code paths, a rate from this one is
circuit-sampled and a rate from `dem_sampling` is model-sampled; the tests pin the
two to each other rather than letting either stand for the other.

## Change and verify

Use [repetition.py](repetition.py) for experiment composition,
[decoders.py](decoders.py) for decoding, [decoding_graph.py](decoding_graph.py)
and [matching.py](matching.py) for the detector-error-model decoder,
[sampling.py](sampling.py) for sampling detection events from a memory circuit,
[noise.py](noise.py) for code-specific noise profiles, [codes.py](codes.py) for
code records, [circuit.py](circuit.py) for detector and observable layouts, and
[types.py](types.py) for records. Run from the repository root:

```bash
python -m pytest tests/qec -q
python tools/check_architecture.py
```

Check syndrome histories, correction actions, and final logical outcomes for
known injected errors. Decoder changes must also cover readout faults and
errors near the final round. Logical suppression or threshold claims require
separate statistical and scaling evidence.

[Feedback modes and model boundaries](IMPLEMENTATION.md) explain the supported
profiles. This reference experiment does not establish general FTQC or
hard-real-time hardware feedback.
