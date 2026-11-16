# Complete diagonal QAOA quantum program

Complete 16-bit 4-layer quantum ansatz with every one of the 137 original cost terms and 8 bound angles. Two nonzero parameter vectors match the original and independent Circuit over all complex amplitudes, including identity phase and logical bit order. Complete Notebook optimization workflow remains pending.

Existing static Observable evolution, bounded quantum helpers, initial Hadamards, RX and static loops express exp(-i gamma H) followed by RX(beta). No new special-purpose synthesis API, production lowering or budget increase. Circuit references independently transform bounded Boolean spin truth tables; source data and native Observable coefficients are checked against frozen original literals. Zero workspace, both objective instruction sequences and actual resource refusals are verified. No user example imports Qmod.

Full complex128 amplitudes use absolute tolerance 1e-12, without phase alignment, thresholding, term truncation or reduced widths/layers. The first parameter vector produces 1584 static IR instructions and depth 348; these are not hardware-basis costs or optimality claims.

[README](README.md) gives runnable native and Circuit commands. Notebook data preparation and optimization require separate workflow acceptance.
