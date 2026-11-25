# Original Hubbard GQSP degree-127 ancillary boundary

The complete original textbook-QPE test includes a decomposition regression for
`HamiltonianSimulationByGQSP` on the 2-by-2 Hubbard model, hopping and interaction
both one, time five and precision `1e-7`. This batch verifies that exact case and
its complete direct PREPARE, SELECT, controlled reflection and GQSP dependencies.
Reference: unmodified Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

## Result and source limitation

The existing native operation constructs successfully at the original parameters:
normalization 20, degree 127 and 255 signal rotations. A generic precision or
construction refusal is no longer the remaining issue. The original
`assert_valid_bloq_decomposition(TextbookQPE(...))` assertion also passes.

On a normalized complex superposition of three target basis states, with signal,
selection and all ancillary bits initially zero, the complete native and source
channels before the final original Free agree in absolute complex amplitudes.
Maximum discrepancy is below `1e-14`. However, the two outer PREPARE junk bits
have nonzero probability approximately **0.3289881219704**. The complete state
has norm one. Projection onto zero junk retains squared norm approximately
**0.6710118780290**; no normalization is applied to conceal that loss.

The original graph explicitly applies two clean Free bits after
PREPARE/GQSP/unprepare. Its Free tensor is a zero bra. Structural decomposition
validation therefore does not certify physical auxiliary cleanup or unitarity
of the resulting 17-bit root. This is a concrete counterexample at the original
parameters, rather than extrapolation from the earlier degree-one example.

Native `HubbardHamiltonianGQSP` retains the two outer junk bits explicitly.
They are logical output storage, not clean reusable workspace. Omitting them
is rejected. They cannot be silently supplied as supposedly restored QPE work,
discarded, reset or projected to make a 17-bit unitary callback. Neither original
source nor native algorithm is changed to repair the reference in this batch.
Q-0634 remains open for its complete native QPE integration obligations; this
counterexample records the exact source boundary rather than claiming success.

## Ownership and exact channel proof

This is `single_device_fast_path` scientific verification. Existing Algorithms
owns composition, Compiler owns instruction transformation, Ecosystem imports
original gates, and Runtime/Simulation retain execution ownership. No production
owner, public entry, constructor contract or serialization vocabulary changes.
The existing `circuit.append(op)` path is checked on the actual 28-bit operation.

Canonical logical order is signal, the eight named selection bits, eight target
bits, and two retained outer junk bits. All words are MSB-first. Two separate
reflection junk bits and seven clean workspace bits are restored between
children. Seven measurement slots are reused after each child's writes.

The proof uses actual emitted children, not a substituted Hamiltonian matrix:

- The complete 10-bit PREPARE and inverse operators are evaluated on all 1,024
  basis columns. Native and independently captured original Always(PREPARE)
  streams agree, including every complex phase and private-work cleanup.
- Native and original SELECT phased permutations match on all 65,536 logical
  inputs. Both native controlled polarities and both independently emitted
  directions are checked on all 131,072 inputs each. Inactive sectors are exact
  identities; all private work is zero.
- The shared all-history And certificate now also tracks monomial complex phases.
  Its former positive-permutation entry retains its positive-phase assertion.
  Actual dense primitive execution on every small basis input and both raw
  measurement outcomes independently validates Y, CY, diagonal phases, swaps,
  exact CSwap and the existing And/erase blocks. Unknown gates, conditional
  atoms and invalid erase predicates are refused.
- The complete source selection-only reflection is formed from the verified
  PREPARE operator. Its action on all clean-junk inputs restores reflection
  junk, with leakage amplitude below `3e-18`. Every controlled zero-mark column
  is checked from actual primitive instructions and every erase branch.
- All 255 emitted native signal rotation matrices match the actual original
  rotation matrices, with maximum discrepancy below `7e-16`. The source's actual
  509-operation boxed GQSP schedule determines all control polarities, operands
  and directions. Literal child concatenation matches the complete operation
  appended to a native DynamicCircuit.
- Composing these independently verified full complex channels produces the
  complete retained-junk state. This avoids allocating a dense 28-bit state
  vector while preserving coherence, phase, all work-zero proofs and every
  raw history. It is a test certificate, not a new execution backend.

Every original PREPARE, SELECT, GQSP, reflection, Free and textbook-QPE source
and relevant test blob used here is checked against the independent locked
manifest. Qualtran 0.7.0 and native Kickmix 0.1 Simulator are verified before
reference tests. Source code is not edited or replaced by the native reference.

## Resources and verification

Actual emitted instruction inventory, all-history measurement exponent, physical
storage and the original QEC model/peak are recorded separately in JUnit.
The actual body contains 258,411 emitted instructions, including 60,038 T/Tdag,
94,874 CX, 33,828 H, 7,880 measurements and 7,880 resets; the positive-scalar
child proof covers all `2^7880` histories without attempting floating-point
representation of each tiny unnormalized branch. The 17 source-visible bits,
two retained outer junk bits, two reflection junk bits and seven work bits total
28 physical bits. Zero junk projection is not counted as workspace cleanup.
The original five-child resource model remains a model, independently checked
against the native declared call graph. Its QEC estimate is 4,074 CSwap, 7,370
And, 23,136 Clifford, 2,805 rotations and 7,370 measurements, with peak 28.
The measured instruction inventory is not relabeled to equal these model units.

Receipts use `g2-hubbard-gqsp-full-degree-*20261125` in the reference workspace.
The concentrated regression additionally checks old positive-permutation users
in QROAM, QFT, compare/exchange, sorting and bounded increment. Original-source
regressions and the prior local integration evidence are reused; this batch
makes no whole-repository, GPU or performance claim. Pending proposals and
ReadTheDocs disposition remain unchanged.

Concentrated dependency and existing-certificate regression: 197 passed.
Independent Cirq 1.6 run: 27 passed. Complete changed-file strict typing,
Black/Ruff, Integration scope, language, links, architecture and IR/API checks
are recorded with the batch. A final focused run covers the additional full
source PREPARE inverse, controlled SELECT inverse columns and clean Free
assertions. The unchanged 720-diagnostic dependency baseline remains explicit.
