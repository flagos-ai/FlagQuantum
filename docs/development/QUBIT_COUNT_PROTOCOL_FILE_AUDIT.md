# Logical qubit-count protocol source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Source file | Frozen Git blob | Native disposition |
| --- | --- | --- |
| `resource_counting/_qubit_counts.py` | `723e06b64ef3b589abb36538e19b79369f1bb8f8` | Connection peak estimation, recursive child models, and signature/callee fallback share existing Compiler owners |
| `resource_counting/_qubit_counts_test.py` | `6f840fc8697cd7f2b649c63540a7b7e263abef3f` | Seven original non-notebook tests run unchanged. The paired native coverage includes all three allocation strategies, cache values, and disconnected components |
| `resource_counting/qubit_counts.ipynb` | `c23db8d556b786164ae87aa9a915119729b1de4a` | Native `qubit_count_protocol.ipynb` preserves the eight-cell lesson and all five code topics |

The native counterpart to `_cbloq_max_width` schedules each weakly
connected component independently. At each step it adds live bystanders
to the callee width and checks the outgoing connection width. Allocation
and free use the shared greedy ordering. The typed Program query uses
the same SSA lifetime connections as register scheduling. It validates
the Program before estimating, applies source-compatible connection
bitsizes, and recurses through finite typed call bodies. At primitive or
mixed typed bodies it falls back to the signature and declared callees.
This fallback is a lower-bound model for bodies with private work.
The generic query carries static overrides and caches, and accepts
symbolic width connections through the declared decomposition owner.

The pinned interior example has a `2*n` signature, a `3*n` connected
peak, and a child width cache of `n` for Allocate and Free, `2*n` for
Swap, and two for the bit swap. The independent and serial source
examples both peak at one. Explicit reuse, per-call allocation, and
factored allocation each give a peak of 11 at width 10.

| Interior-allocation evidence | Result |
| --- | --- |
| Signature model | `2*n` |
| Declared and decomposed peak models | `3*n` |
| Source QEC model | `2*n` Clifford |
| Bound native primitive stream | `2*n` SWAP, one allocate/free lifetime, zero measurements |

The retained native interior tests compare gate order, operands, outputs,
and allocation events against the source. Full complex amplitudes are
checked on the clean-output input subspace. The allocated word is zero
on return. The source rejects freeing the word when the output input
port started nonzero, and the native witness leaves nonzero work for
that inadmissible state. The three repeated allocation fixtures separately
verify their gate streams and restoration.

The original and native notebooks run in separate kernels. Each displays
two SVGs and one register-score PNG. The native score binds width two and
the symbolic interface and resource graph retain `n`. The symbolic
decomposition model and the finite typed estimator agree after binding.
Model queries emit zero instructions.
