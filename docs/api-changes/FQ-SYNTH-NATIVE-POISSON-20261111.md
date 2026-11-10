# Native discrete Poisson transform and HHL composition

Status: native/frozen-source full-state probe passed; independent Circuit and
formal acceptance pending. Coverage starts at 204/216.

Preserve the frozen 64-amplitude spatial input, two three-bit spatial axes,
Type-II sine/cosine transforms, sixteen ordered diagonal Pauli terms including
both identity terms, scaling 0.101321183642338, six-bit phase estimation, all
64 inverse-rotation coefficients, inverse QPE and inverse spatial transforms.
Do not substitute an ideal linear solve, change the rounded rotation table,
reduce spatial width or erase residual phase-estimation amplitudes.

Reuse public amplitude preparation, bounded unitary on static slices,
suzuki_trotter, qpe, select_rotation and within/apply composition. Native
qst_type2/qct_type2 require complete register views; the six-bit spatial array
may instead use the mathematical 8x8 Type-II kernels on its two three-bit
slices. These are input-independent transform primitives, not compiled final
solution states or a hidden Qmod loader. Retain the original low-word-first
axis assignment and prove the complete bit order.

Native qpe requires an all-fractional numeric phase port. The frozen source
uses its phase word as an unsigned integer table index. A native six-fractional
annotation preserves the same six bits and table lookup; document that users
must interpret this port by its raw word for the retained inversion table.
Do not claim its native numeric values have the original integer scale.
All phase residuals remain visible; compiler work must return to zero.

Independent Circuit acceptance must derive spatial transforms independently,
prepare the full input with a different construction, assemble diagonal
controlled evolution and analytic inverse DFT, apply the exact rounded
rotation table and undo in reverse order. Compare all 8192 complex amplitudes
against native and frozen source, preserving absolute phase. Include two
additional normalized nonzero signed inputs, primitive transform/inverse tests,
controlled identity-phase interference, register/phase indexing, alias and
budget refusals. Preserve original resource records and report native overhead.
No production export, API, effect cap or ownership rule changes are planned.

After acceptance run fresh 216-original plus approved-six replay, archive
immutable evidence and update catalog/ledger. Notebook workflows remain separate
at 24/213. Classification is `single_device_fast_path`. No push, integration PR
or full remote CI before all 216 native quantum programs have independent
acceptance.
