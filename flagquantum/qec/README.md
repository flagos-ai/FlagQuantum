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

## Ten-minute change path

1. Add or modify a QEC-owned code, decoder, or result rule.
2. Keep generic dynamic-program changes in Compiler or Runtime.
3. Add a scenario test under `tests/qec`.
4. Run:

   ```bash
   python -m pytest tests/qec -q
   python tools/check_architecture.py
   ```

The reference accepts deterministic errors only. No stochastic-noise,
measurement-error, logical-error-suppression, threshold, fault-tolerance,
realtime-hardware, or performance claim follows from the workflow.
