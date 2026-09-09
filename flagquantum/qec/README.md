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
`compiled_lookup` performs immediate reference feedback; the separate
`offline_pauli_frame` mode leaves the data uncorrected until a decoder-produced
frame is applied to final readout. Executed feedback and decoder advice are
reported separately. A replaceable `Decoder` consumes the complete syndrome
history after execution; it is not yet called in a real-time control loop. The
namespace is not exported from the stable `flagquantum` root API.

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

The stochastic profile is limited to independent bit flips and independent
readout confusion. No general Kraus/correlated/timing-noise,
logical-error-suppression, threshold, fault-tolerance, realtime-hardware, or
performance claim follows from the workflow.
