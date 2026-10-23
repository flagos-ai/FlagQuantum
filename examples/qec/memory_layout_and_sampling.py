"""Read memory layouts, sample with the Stim extra, and run existing feedback modes."""

import pickle

from flagquantum.qec import (
    ErrorEvent,
    ErrorSchedule,
    MemoryCircuit,
    PhenomenologicalNoise,
    RepetitionCode,
    build_memory_circuit,
    decoder_context_from_memory_circuit,
    run_repetition_memory_experiment,
    sample_memory_circuit,
    sample_memory_measurements,
)
from flagquantum.qec.circuits import MemoryCircuit as CircuitRecord
from flagquantum.qec.execution.context import DecoderContext

memory = build_memory_circuit(RepetitionCode(), rounds=3)
context = decoder_context_from_memory_circuit(
    memory, noise=PhenomenologicalNoise(data_flip=0.125, measurement_flip=0.25)
)
inputs = context.full_component()
print("Same circuit class:", MemoryCircuit is CircuitRecord)
print("Context class:", type(context) is DecoderContext)
print(
    "Measurements, detectors, observables:",
    context.num_measurements(),
    len(memory.detectors),
    len(memory.observables),
)
print("Detector map shape:", tuple(inputs.measurement_to_detectors.dense().shape))
print("Pickle class:", type(pickle.loads(pickle.dumps(memory))).__module__)
measurements = sample_memory_measurements(
    memory, noise=PhenomenologicalNoise(), shots=4, seed=53
)
events = sample_memory_circuit(memory, noise=PhenomenologicalNoise(), shots=4, seed=53)
print("Noiseless measurements:", tuple(measurements.outcomes.shape))
print(
    "Detection events, observable flips:",
    int(events.detectors.sum()),
    int(events.observables.sum()),
)
for mode in ("compiled_lookup", "offline_pauli_frame", "runtime_decoder"):
    result = run_repetition_memory_experiment(
        rounds=3,
        shots=2,
        seed=53,
        strategy="trajectory",
        feedback_mode=mode,
        error_schedule=ErrorSchedule((ErrorEvent(1, 1),)),
    )
    print(mode, result.logical_error_rate, result.shot_records[0].decoded_data_bits)
