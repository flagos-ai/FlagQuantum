# Quantum error correction

This domain owns quantum-error-correction concepts and workflows: code
definitions, syndrome and detection-event records, decoder contracts,
correction decisions, and logical-result analysis.

It does not define generic circuit, compiler, runtime, simulation, noise, or
provider semantics. Dynamic control remains in Compiler and Runtime; QEC
workflows compose those capabilities. Numerical kernels remain in Simulation,
and hardware feedback contracts remain in Remote or a future realtime Runtime
domain.

The first experimental profile contains a three-data-qubit repetition code, a
bounded deterministic X-error schedule, a two-bit-syndrome lookup decoder, and
a fixed-round memory experiment using the private bounded hybrid compiler.
`compiled_lookup` performs immediate reference feedback inside the lowered
circuit; the separate `offline_pauli_frame` mode leaves the data uncorrected
until a decoder-produced frame is applied after execution. Two local Runtime
modes call a replaceable `StreamingDecoder` after every syndrome round and
either apply a physical X (`runtime_decoder`) or update an X Pauli frame
(`runtime_pauli_frame`). The frame adjusts later syndrome interpretation and
final readout without changing the quantum state. Shot traces retain true and
observed bits, actions, and frame evolution. This is not a hard-real-time or
provider feedback contract. The namespace is not exported from the stable
`flagquantum` root API.

`RepetitionTemporalDecoder` adds a bounded measurement-error-aware policy. It
requires the same non-zero syndrome in two consecutive rounds before acting.
An isolated readout excursion therefore clears with a paired detection event
and produces no correction. The extra evidence round means errors first seen
in the final round remain visible but unconfirmed. The corresponding modes are
`runtime_temporal_decoder` and `runtime_temporal_pauli_frame`.

`RepetitionNoiseProfile` maps code-specific circuit locations onto the existing
backend-neutral `NoiseModel`: a data bit-flip channel is sampled after matching
parity-check CNOTs, syndrome readout confusion applies to ancilla measurements,
and optional final readout confusion applies to data wires. Because the middle
data wire participates in two check CNOTs per round while each edge wire
participates in one, its configured channel has two opportunities per round.
Finite-shot sweeps report observations and event counts; they are not threshold
or logical-suppression evidence.

## Ten-minute change path

1. Add or modify a QEC-owned code, decoder, or result rule.
2. Keep generic dynamic-program changes in Compiler or Runtime.
3. Add a scenario test under `tests/qec`.
4. Run:

   ```bash
   python -m pytest tests/qec -q
   python tools/check_architecture.py
   ```

Runtime decoder feedback is trajectory-only; explicit batched feedback fails
closed. The stochastic profile is limited to independent bit flips and
independent readout confusion. No general Kraus/correlated/timing-noise,
maximum-likelihood or general measurement-error-tolerant decoder,
logical-error-suppression, threshold, fault-tolerance, realtime-hardware, or
performance claim follows from the workflow.

## Code-independent records

`pauli.py`, `codes.py`, and `circuit.py` add a code-independent layer beside the
frozen repetition profile. A `Pauli` is a phase-free operator over arbitrary wire
indices; a `StabilizerCode` is a value that declares its distance, wire layout,
checks, stabilizers, and logical observables; `build_memory_circuit` turns a code
and a round count into circuit source plus a detector layout and an observable
layout.

Detector semantics are fixed. A detector is a measurement parity that is
deterministic in the noiseless circuit. A `rounds`-round memory experiment
declares one detector per check per round, comparing that round against its
predecessor, where the first round is compared against the known all-zero prior
state, plus one detector per check comparing the final syndrome round against the
terminal data readout. That grammar is `len(checks) * (rounds + 1)` detectors, and
for a distance-`d` repetition code, whose `len(checks)` is `d - 1`, it equals
`(d - 1) * (rounds + 1)`. Logical failure is the parity of a declared logical
observable. The detector count and every detector and observable reference are
validated against the code's declared checks, wires, logical observables, and the
configured round count, so a hand-built layout cannot silently disagree with the
code it is paired with.

This layer is additive. The frozen repetition types, the two decoder protocols,
three reference decoders, and both existing workflows keep their current
signatures and behavior, and the seven names pinned by
`contracts/hybrid-compilation-private-v0-candidate.json` are unchanged. The
`RepetitionCode` at `distance=3` reproduces the frozen circuit instruction for
instruction, which is asserted by
`tests/qec/test_memory_circuit_execution.py`.

The frozen profile defines logical failure as the majority of its three
frame-corrected data bits. That is a profile-specific decision rule rather than a
linear logical observable, so general-layer failure counts are not claimed to
equal frozen-profile failure counts at `distance=3`. This layer does not change
the frozen profile's arithmetic.

The layer declares codes and detectors only. It does not build a detector error
model, decode, sample evidence, or make any threshold, logical-suppression,
real-time, or fault-tolerance claim.

One representational boundary is explicit and enforced. `CodeCheck` requires every
CNOT to control a data wire and target the check's ancilla, so it describes only
Z-type checks measured with a Z-basis ancilla; an X-type stabilizer is rejected by
validation with "check stabilizer must be Z-type". An X-type check couples the
ancilla the other way and is not representable here, even though the `Pauli`
record and the check's `stabilizer` field are basis-agnostic. This matches the
repetition code, which detects bit flips; a code family needing X-type checks
requires this record to grow before it can be described.
