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
The junk stays allocated after a source-compatible step. The original
symbolic qubit-peak query raises when reflection tries to decide whether
the symbolic junk width is positive.

For the one-site Ising fixture and a signed two-target Pauli fixture,
append a step and its inverse with the same signal junk and scratch
bindings. Both orders recover the initial complex state with all work
zero. Each of the 64 first-step histories produces the same unnormalized
intermediate state, including its occupied junk sectors. Composing it
with each of the 64 inverse histories proves recovery for all 4,096
history pairs in each order for each fixture. This evidence permits
uncomputation by reversal on the retained registers. The locked single
step can still leave the preparation junk occupied.

The native symbolic peak specializes to the concrete resource schedule at
a fixed number of Hamiltonian terms. Dense PREPARE takes the maximum of
its signature, uniform preparation with live alias junk, QROM with live
probability and comparator output, and the equal-width comparator with
live selection and alternate registers. The walk carries the Hamiltonian
target through PREPARE and through the zero mark, whose live width also
includes the alias junk. The SELECT peak comes from its concrete
declaration. The existing symbolic gate model retains the locked
shape-only call graph.

The native expressions are checked at 1, 2, 3, 4, 8, and 16 sites and four
precisions per site. Dense PREPARE matches the locked concrete
`QubitCount`, and the whole-walk peak matches both the locked concrete
`QubitCount` and the native concrete estimate. These are model widths.
The source symbolic reflection and walk failure remains reproducible.

## External Hamiltonian boundary

The source consumes a Cirq `PauliSum`. The differential test maps its
qubit order to native integer labels before native term merging. Three
fixtures check mixed grid and named labels, merged duplicate terms,
an entirely cancelled wire, signed words and identity contributions.
The concrete alias tables and QEC models match, and the three symbolic
peaks specialize to 26, 25 and 26 logical qubits, respectively.

A four-row mixed-label case also compares the full complex output of
the native circuit against the locked PREPARE and SELECT tensors and
the locked two-bit zero reflection. Four representative measurement
histories per direction pass, including all-zero, all-one and both
alternating histories. All sixteen checked history pairs recover the
input and clear work in each direction. The model gives a peak of
16 logical qubits and 68 T equivalents. The bound circuit uses 17 wires
and emits 80 T/T† and 13 measurements.

The source admits a one-product declaration whose zero-width SELECT
rejects the resource query. The native walk refuses that shape at
construction. For complex coefficients, the source takes the
absolute value of the real part for alias weights while retaining the
complex coefficient in the dense SELECT word. The native real walk
rejects this non-Hermitian input. The differential conversion lives
in tests and the native factory receives a FlagQuantum Hamiltonian.

Q-0357, Q-0358 and Q-0360 close the native Hamiltonian and test coverage.
Q-0359 remains open for one-step clean preparation junk, the source's
symbolic width and singleton failures, its mixed-sign norm estimate,
and direct external object binding.


## Faithful complete-source acceptance (2026-11-28)

The approved faithful-migration closure rule supersedes historical open status
for Q-0359. Complete source review covers all four functions: eigenphase to
coefficient precision, the signed-sum Pauli norm formula, generic Pauli walk
construction and the periodic Ising wrapper with its original default strengths.
Existing Hamiltonian/alias/SELECT/reflection owners provide the implementation.
Concrete and symbolic precision/norm paths, ordered merged Pauli words, label
mapping and source normalization are retained without adding another backend.

The signed norm estimate is the original formula, including its unsafe bound
and negative-radicand witnesses. The native real-Hamiltonian operation refuses
a nonreal estimate and accepts an explicit bound. It does not claim support for
source non-Hermitian coefficient admission. Singleton/zero-selection source
failures and symbolic peak failure remain recorded, separately from native
concrete-specialized symbolic resource predictions.

The original reflection can leave signal junk occupied. Native joint execution
retains that environment as caller-owned data; only actual private scratch is
required zero. A single step is not reported as a clean selection-only unitary.
Existing complete small histories, full complex joint amplitudes, inverse
restoration with the same signal environment and actual four-site source
physical capture remain authoritative. Source ABI refusal prevents silently
discarding those ports. Larger explicit-state witnesses are not full dense
operators or brute-force enumeration of every large-example history.

The current concentrated Ising/walk/constructor batch passes 186 cases;
57 independent Cirq 1.6 context/Ising cases also pass. All 88 reviewed locked
source/dependency/asset blobs are unchanged. Existing strict production,
32 local integration and real Kickmix Simulator evidence is reused because no
Python implementation changes in this acceptance entry. Direct generic walk
chemistry dependency ownership remains open under Q-0658, not counted here.
