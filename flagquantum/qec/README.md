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
two-bit-syndrome lookup decoder, and a fixed-round memory experiment using the
private bounded hybrid compiler. The compiled program owns the reference
lookup feedback. A replaceable `Decoder` currently interprets the recorded
syndromes after execution; it is not yet called in a real-time control loop.
The namespace is not exported from the stable `flagquantum` root API.

## Ten-minute change path

1. Add or modify a QEC-owned code, decoder, or result rule.
2. Keep generic dynamic-program changes in Compiler or Runtime.
3. Add a scenario test under `tests/qec`.
4. Run:

   ```bash
   python -m pytest tests/qec -q
   python tools/check_architecture.py
   ```

No logical-error-rate, threshold, fault-tolerance, realtime-hardware, or
performance claim follows from the reference workflow.
