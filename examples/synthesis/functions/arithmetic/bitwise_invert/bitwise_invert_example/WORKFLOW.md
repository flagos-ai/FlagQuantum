# Finite-width inversion workflow

The reviewed Notebook declares x = 6, computes y = ~x, synthesizes and samples.
The native qfunc keeps that high-level logic: the inferred three-bit encoding
complements 110 to 001, so named outputs are (6, 1). The user writes no gates,
wire indices or bit-string decoding.

Run main.py as a module to synthesize, execute and draw 1000 seeded local samples
from executed named probabilities. This replaces the Notebook cloud sampler; it
does not claim a hardware execution. Independent Circuit tests compare absolute
complex amplitudes, bit order and clean workspace before sampling. Additional
coherent tests verify every three-bit input and unequal-width word operations.
Classical-only ~2 is refused because it has no quantum encoding width.
