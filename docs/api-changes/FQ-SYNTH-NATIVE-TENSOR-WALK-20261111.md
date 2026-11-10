# Native tensor-hypercontraction zero-entry walk

Status: semantic contract; corpus acceptance remains 206/216.
Classification: `single_device_fast_path`.

Preserve the original four-bit clock, all 71 ELL bits, all eight PSI bits,
sixteen rounded clock amplitudes, two spin Hadamards and inverse QFT. The source
initializes ELL to zero. Its select is identity there because the succ guard is
zero and the compute/uncompute conjugations never modify succ. Its reflection
is identity because prepare† maps zero ELL into the succ=1 subspace: the reverse
last preparation marker sees mu=nu=0 and flips succ. Later inverse gates before
that marker preserve succ. The reverse later preparation stage preserves the
zero addresses and succ, even when its QROM modifies other fields. Therefore
all walk calls preserve ELL=0 for arbitrary clock/spin amplitudes, and every
classical clock control work bit returns to zero.

Native code should expose this initialized-subspace specialization explicitly,
retain all outputs, and compose clock preparation, the proved identity walk and
inverse QFT without loading source, compiler proofs or tests. It does not offer
nonzero ELL inputs or claim a general tensor-hypercontraction walk unitary.
This is exact reduction of this frozen program, not a replacement chemistry
simulation or a completed Notebook workflow.

Independent acceptance must include a mathematical proof of preparation
orthogonality and select guard invariance, a separate complete finite clock
control trace, all occupied-support complex amplitudes with total probability
one, zero inactive fields, no MPS truncation and no dense 2**83 allocation.
Compare default plus two nonzero signed clock vectors against frozen source and
an independent Circuit; test nonzero coherent clock phases and changed spin
preparation, memory/qubit/work budgets and source proof mutations. Preserve
original bytes and resource baseline. Native exact interval MPS reuses the
Runtime bond proof; admission changes must be explicitly pinned in evidence.

After passing, run fresh 216-original and approved-six replay, compare all
original outcomes/resources, archive immutable evidence and update catalog and
ledger. Notebook/Python workflow progress remains separate. No push, integration
PR or full remote CI before all 216 native quantum programs pass acceptance.
