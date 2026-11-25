# Retained native Pauli evolution

Retained 2-qubit, 3-term quantum program with one first-order Lie product step. This is not exact matrix exponentiation. Complete companion Notebook workflow remains pending.

`hamiltonian.json` is numeric problem data, not a quantum source file. Native code uses the immutable FlagQuantum Observable and existing checked Pauli lowering. The independent Circuit uses inverse-CDF sampling rather than the production choices sampler, and uses S-dagger/H basis changes with an opposite parity anchor. Complete complex128 amplitudes include absolute global phase. No phases are aligned away.

The source-profile exponentiate call means a single first-order product-formula step; the native suzuki_trotter call preserves that definition. It does not claim exact exponential accuracy.

[README](README.md) gives runnable native and Circuit commands.
