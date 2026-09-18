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
observable.

The detector count is validated against the code's declared checks and the
configured round count, and every reference is validated against the code's
declared wires: a detector's terminal readouts must name declared data wires and
its syndrome measurements must name declared ancilla wires inside the configured
rounds, and an observable's readout must name declared data wires and match the
code's declared logical operators. The validation is membership-based. It does not
check that a detector names the *same* check in each round, that a terminal
detector's data wires are the support of the check it belongs to, or that the
`source` text is the program those layouts describe, so a hand-built layout can
still be semantically wrong while satisfying every check.

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

The layer declares codes, detectors, and detector error models. It does not
decode, and it makes no threshold, logical-suppression, real-time, or
fault-tolerance claim.

`dem.py` adds a detector error model over that layer, and `PhenomenologicalNoise`
in `noise.py` is the noise record it consumes. A `DemError` is one independent
physical error mechanism, named by the detectors and the logical observables it
flips; `DetectorErrorModel` is a set of them over a fixed detector and observable
count, with its parity matrices, its exact marginal rates, seeded sampling, and
stim text interchange. `from_memory_circuit` builds the model of a memory circuit
under a `PhenomenologicalNoise`: one mechanism per data wire per round, and one
per check measurement per round, minus the locations whose probability is zero,
with the mechanisms that flip the same detectors and observables merged. The
model is exact for Pauli noise in the reference gate set.

Construction is exact and does not sample. Each mechanism's signature comes from
one forced execution: the single error is injected into the circuit source, the
source is lowered and executed twice, and the two shots must agree before the
signature is read off the detector and observable layouts. That two-shot
determinism assertion is both the fail-closed check for a mechanism that is not a
Pauli mechanism in the reference gate set and the evidence that the signature is
not sampled. The model is defined for Pauli noise only: a non-Pauli channel is
refused with a stated reason rather than approximated.

The model is built on the memory circuit without in-circuit feedback, because it
describes the noise-to-detection mapping that a decoder inverts. The frozen
profile's `compiled_lookup` mode bakes immediate reference feedback into the
frozen profile's own private source rather than a `MemoryCircuit`, and the
forced-error engine drives `MemoryCircuit` sources, so mechanical limits leave
that compiled-feedback path unmodelled.

The detector-rate evidence is a cross-check, not an independent derivation.
Phenomenological noise is the primary evidence: the model's exact marginal rates
are compared at distance three and distance five against rates sampled from an
injected-shot circuit simulator. The comparison shares the injection helper with
construction, so it does not independently re-derive the signatures; what it
tests is that mechanisms compose by XOR in the simulator and that merging
identical signatures yields the right marginals. The one genuinely independent
check available — parsing the emitted text with the real `stim` package — was
run at developer time and is not a committed test, because `stim` is not a
dependency of this repository.

The stim interchange is partial: this reader accepts a subset of stim output, and
which subset is fixed by what the text declares rather than by whether stim
produced it. `to_stim_text()` emits valid stim text, verified against stim
1.16.0. `from_stim_text()` reads the format `to_stim_text()` emits and
hand-written text in that style, not arbitrary stim output: the declarations must
be complete and consecutive from zero, `#` comments are not accepted, and the
model shape comes from the declarations alone. `shift_detectors` is refused
outright.

In the developer-time sweep -- 96 detector error models from repetition-code and
rotated-surface-code memory circuits, distances three and five, one to three
rounds, noisy and noise-free, with and without flattening the circuit first --
`shift_detectors` accounted for 32 refusals and the undeclared observable index
for the other 32. All 48 that carried an error were refused, and the 32 accepted
carried none.

A second sweep of 240 single-error circuits -- one `X_ERROR` on each data wire,
the same two families, distances and rounds -- refused 170 of them: 160 by the
`shift_detectors` rule and 10 because an error index fell outside the declared
shape. The 70 that satisfied both conditions were all accepted, and 32 of those
carried an error. So the conditions are two, and both are needed: of the 210 of
those circuits whose errors avoided the observable, the 140 whose rounds repeat
were refused anyway. Both sweeps were also checked for `repeat` blocks -- none of
the 336 texts contained one, so no refusal above came from that rule.

Where the rates were checked against stim rather than restated: the 32 accepted
DEMs that carry an error match stim's own compiled sampler to within 0.36
standard deviations over 20000 shots. On the edited route -- strip the observable
instruction from a noisy circuit and the flattened text is accepted, errors and
all -- the agreement is 2.61 standard deviations over the same shot count.

Not covered by either sweep: decomposed or approximately-disjoint errors, gauge
detectors, repeat blocks, colour codes, distances above five, and hand-written
text.

`DemSample` carries tensors and defines content equality, and it is deliberately
unhashable: it must not be used as a set member or a dict key.

The scope refusals hold here as they do for the rest of the layer. No threshold
claim, no logical-suppression claim, and no real-time or hardware-feedback claim
follows from a detector error model or from any rate it reports. A rate the model
samples is a DEM-sampled rate, and it is labelled as one wherever it is reported.

`README.md` is deliberately not updated for the detector error model: the Stage 5
documentation sweep owns it.

One representational boundary is explicit and enforced. `CodeCheck` requires every
CNOT to control a data wire and target the check's ancilla, so it describes only
Z-type checks measured with a Z-basis ancilla; an X-type stabilizer is rejected by
validation with "check stabilizer must be Z-type". An X-type check couples the
ancilla the other way and is not representable here, even though the `Pauli`
record and the check's `stabilizer` field are basis-agnostic. This matches the
repetition code, which detects bit flips; a code family needing X-type checks
requires this record to grow before it can be described.
