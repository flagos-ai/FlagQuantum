# Periodic Ising Hamiltonian and walk audit

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under `qualtran/bloqs/chemistry/ising`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0357 | `__init__.py` | `166d3e526f6b573dc6ece8fe162eb2df107ff68c` |
| Q-0358 | `hamiltonian.py` | `e71db3d18472e814c6b808b69bf9fa69dfcf7442` |
| Q-0359 | `walk_operator.py` | `7c237150ff3e0b46344603e3ff75f20e7a6806ed` |
| Q-0360 | `walk_operator_test.py` | `000666430bda2a1a12f6717a266c777b2ffb06a4` |

## Native owners

`ising_hamiltonian` owns the unmerged periodic terms, merged Hamiltonian,
signed LCU coefficients and Ising norm bound. Raw terms retain the order
of the supplied lattice labels. The merged Hamiltonian places every
nonzero ZZ term before the X terms, including the collapsed one-site bond
and the doubled two-site bond. The coefficient tuple follows this actual
source order. The original coefficient helper's prose calls it interleaved.
The native `Hamiltonian` class requires at least one term. The separate
coefficient query still returns an empty tuple for an all-zero chain.

`ising_walk` constructs the chain through the common Pauli walk factory.
The factory owns signed SELECT words, absolute alias weights, sorted
target labels and eigenphase-to-coefficient precision. Symbolic plans
accept a symbolic norm bound as well as symbolic eigenphase precision.
The native walk uses the same SELECT, PREPARE and reflection owners through
`circuit.append(op)`. The signal junk is separately supplied and kept live
by the caller. Measurement feedback uses the compiler-owned work wires.

## Evidence and resource accounting

- Both unmodified original tests passed. Native tests compare periodic
  terms on reordered labels, coefficient ordering, normalized alias data,
  the original formula, and complete concrete and symbolic gate categories.
  Concrete cases cover one to four sites.
- A native one-site chain with `j_zz=3` and `gamma_x=-3` compares every
  output complex amplitude in the logical and signal-junk registers with
  the source SELECT and reflection matrices. All 64 measurement histories
  pass in each direction and the compiler scratch is zero.
- That source model gives 32 T equivalents and a peak of 13 logical
  qubits. The native instruction stream emits 38 T/T† and six
  measurements in that fixture.

## Source boundaries

The generic norm helper uses the source's signed coefficient sum.
For `4X-Z`, the locked formula gives `1` while the operator norm is
`sqrt(17)`. Some mixed-sign cases produce a negative radicand. The
native query raises for a nonreal estimate and the walk accepts an
explicit valid bound.

The locked alias reflection can leave its allocated preparation junk
nonzero. The exact amplitude comparison includes every sector of that junk.
The junk stays allocated after a

source-compatible step. The original symbolic qubit-peak query also
raises when it reaches the symbolic preparation junk. Gate estimation
is checked independently.

Q-0357, Q-0358 and Q-0360 close the native Hamiltonian and test coverage.
Q-0359 remains open for a recoverable reflection workspace path, symbolic
peak handling and the source's broader external Hamiltonian interface.
