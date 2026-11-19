# Complete matrix-HHL workflow

Classification: `single_device_fast_path`. Preserve the original dense matrix,
normalized [1,2,4,3] vector, two data bits, four unsigned fractional QPE bits and
one indicator. Classical matrix exponential and ordered Pauli decomposition are
recomputed from the source data with the existing PyTorch dependency.

The exact route uses exp(2 pi i A). The approximate route supplies QPE with a
one-argument native callback. For powers 1,2,4,8 it calls first-order
suzuki_trotter with time -2 pi power and repetitions 4,8,16,24. It does not take
powers of one fixed approximate unitary. Both routes use generated within/apply
inversion after the same reciprocal rotation table. Independent Circuits build
the four controlled evolution matrices and a dense inverse Fourier transform,
apply the reciprocal table and explicitly reverse QPE. Pauli product matrices
use analytic involution exponentials independently of the lowering.

The source's generic HHL callable becomes two visible native root programs.
Root preparation owns the fresh data, phase and indicator; no generic QCallable
or parameterized fresh-Output helper support is claimed. The source preparation
helper remains visible as a standalone native definition.

Actual readout selects phase zero and indicator one, filters amplitudes below
1e-12, rescales by 1/16 and rounds complex parts to five decimals. It computes a
single global correction arg(<A q,b>), applies that angle to the whole vector,
and then takes the real part as the source diagnostic does. The real projection
is explicitly labeled; it is not a physical unitary correction. Both the source
projected fidelity and the complex-state fidelity are reported. Trotter relative
distance must remain below the original ten-percent threshold. Finite-QPE phase
leakage remains reported separately from clean hidden compiler work.

CX/U3 costs come from actual public basis conversion and resource estimation.
Caller-owned operator-exact rules reuse the earlier QSVT example rules without
modifying compiler defaults. Converted full states must retain absolute global
phase. These local resource results make no hardware, speedup, optimization,
Classiq performance or scalability claim. Original source files, importers and
private FlagQuantum APIs are absent from executable native examples.
