# Native oblivious amplitude amplification

The model preserves the original four real data amplitudes, equal Pauli
weights and ordered XX/XZ/ZX/ZZ branches. The first Pauli character addresses
logical data bit0. The low selector bit chooses the Pauli on data bit1 and the
high selector bit chooses the Pauli on data bit0. Because these two choices are
independent, SELECT factors into two one-bit-controlled X-or-Z operations.

The encoding helper loads the uniform block, applies those choices and
unloads it. Starting with X on a data qubit, the active selector branch applies
the exact two-by-two ZX matrix so the net operation is Z. Its controlled phase
is retained; replacing ZX by an approximate or phase-equivalent rotation would
change interference. This uses existing small controlled-unitary support, not
a matrix encoding of the whole OAA algorithm or a source-specific compiler op.

The user main prepares the data, calls the encoding helper, then composes the
block-zero oracle and reflection for a bounded repetition count. The reflection
uses adjoint and within/apply to retain the original -A R0 A-dagger convention.
Only the block is reflected; data and block remain quantum outputs. No source
permutation/Const-check-disabling decorators or arbitrary higher-order callback
API is introduced. Main users supply amplitudes and repetitions without physical
wire numbers or bitstring decoding. Helper code describes logical quantum
choices; the compiler owns gate lowering and resources.

Tests compare the unchanged source, native default and independent dense Pauli
construction without global-phase alignment. All16columns of the complete
one-step data/block operator check nonzero block inputs and bit order. Two
further nonzero real data models run repetitions0/1/2. Named marginals,zero
workspace and actual width/operations/depth/memory budget refusal are checked.
The count0..16 is bounded before synthesis and compiler limits still apply.

Default native resources are4physical bits,zero work,57operations and depth26;
the original adapter uses4bits,zero work,136operations and depth122. These are
measurements for this factorable Pauli model, not general optimization or
performance evidence. The example accepts the quantum program only, not a
complete Notebook workflow or an oblivious-amplification success certificate.
