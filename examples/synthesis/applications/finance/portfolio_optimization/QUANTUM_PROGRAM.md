# Complete portfolio quantum program

Complete 12-bit three-layer quantum program with all covariance and return terms; two nonzero parameter vectors, both width/depth objectives, full source/native/independent Circuit amplitudes and resource rejection checks. Complete Notebook optimization workflow remains pending.

Preserve negative cost phase, gamma-first/beta-last parameters, three 3-bit asset words and slack weights 1,2,3. Existing QNum phase arithmetic, static loops and Pauli-X word evolution are reused. Explicit native fields replace qstruct storage without adding general numeric-array storage. Independent covariance and affine-square Circuit factors use at most six bits; all classical cost inputs are exhaustively checked. User examples never load Qmod.

Complex128 full states are compared without phase alignment or amplitude thresholding. All original bits and three layers are retained. Compiler workspace is zero. Resources are static IR counts, not hardware cost or an optimality claim.

[README](README.md) gives runnable native and Circuit commands. Notebook workflows require separate acceptance.
