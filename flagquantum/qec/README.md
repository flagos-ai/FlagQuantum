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
measurement-error-tolerant temporal decoder, logical-error-suppression,
threshold, fault-tolerance, realtime-hardware, or performance claim follows
from the workflow.
