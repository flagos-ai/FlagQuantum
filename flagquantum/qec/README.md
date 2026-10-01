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

## Change and verify

Use [repetition.py](repetition.py) for the frozen repetition-code experiment
composition, [decoders.py](decoders.py) for decoding, [noise.py](noise.py) for
code-specific noise profiles, and [types.py](types.py) for records.

The code-independent layer sits beside the frozen profile: [pauli.py](pauli.py)
holds the phase-free `Pauli` operator record, [codes.py](codes.py) holds the
`StabilizerCode` protocol and `RepetitionCode`, [circuit.py](circuit.py) derives
detector and logical-observable layouts with `build_memory_circuit`, and
[dem.py](dem.py) builds an exact `DetectorErrorModel` from a `MemoryCircuit` and
a `PhenomenologicalNoise` record. These are unpublished additions to the
experimental `flagquantum.qec` namespace, not Stable Core exports; the design and
authorization are recorded in
[API_CHANGE_PROPOSAL_048_QEC_DETECTOR_ERROR_MODEL.md](../../docs/development/API_CHANGE_PROPOSAL_048_QEC_DETECTOR_ERROR_MODEL.md).

Run from the repository root:

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
