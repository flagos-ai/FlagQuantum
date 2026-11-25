# Native matrix-QPE precision study

Classification: `single_device_fast_path`. Recompute the original NumPy seed1235
8x8 real symmetric matrix and exponentiate it with existing PyTorch. Preserve
all phase precisions1..8, zero-initialized three-bit data and a retained private
data environment. The source computes a shifted matrix but does not exponentiate
it. This native workflow preserves the actual unshifted calculation; eigenphases
wrap modulo one and the original matrix spectrum is not claimed to lie in[0,1).

Eight explicit root signatures use the existing literal numeric-annotation
contract. This does not introduce dynamic fractional annotations. Every root
allocates its original data and phase, calls native QPE and drops owned data.
Drop ends access without resetting or uncomputing. Independent Circuit matrices
retain the complete joint phase/data state. Named numeric phase probabilities
are independently checked against the data-traced marginal, while nonzero data
environment probability remains reported separately from compiler work.

The native study actually converts all eight circuits to an operator-exact
CX/U3 basis and verifies each converted full state, including global phase.
Resource reports are measured local data. The source's stored Qiskit1.0.0
cost arrays remain clearly labeled historical data; the commented optional
installation/transpilation cells are not represented as a fresh Qiskit run.
The plot is a reproducible source-style resource visualization, not a controlled
cross-framework speedup or hardware benchmark. Caller-owned exact conversion
rules reuse the existing QSVT example without changing compiler defaults.
