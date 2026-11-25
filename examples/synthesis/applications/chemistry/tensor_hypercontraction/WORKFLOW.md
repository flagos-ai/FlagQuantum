# Tensor-hypercontraction zero-entry walk

Run from the repository root:

```sh
python -m examples.synthesis.applications.chemistry.tensor_hypercontraction.main
python -m examples.synthesis.applications.chemistry.tensor_hypercontraction.comparisons.circuit
```

This is the exact initialized-subspace specialization of the frozen quantum
program. It retains the four-bit clock, all 71 ELL bits, all eight spin bits,
sixteen rounded clock amplitudes, two spin Hadamards and inverse QFT. ELL fields
are flattened in `ELL_FIELDS` order, with each field's least significant bit
first. Spin bits 0..2 are down, 3..5 are up, 6 is plus1 and 7 is plus2.
`build_program(weights, clock_phase, spin_phase)` admits normalized real clock
vectors and coherent local phase changes without exposing arbitrary ELL input.

Why the initialized walk disappears:

1. SELECT computes private rotation words and spin basis changes, applies phases
   controlled by ELL.succ, then uncomputes. These computations do not change
   succ. On the initialized ELL=0 subspace, succ=0, so SELECT is identity for
   arbitrary spin states.
2. A controlled reflection conjugates a zero-record projector by PREPARE.
   Inverting PREPARE's later stage leaves the two address fields and succ zero:
   the last swap exchanges zero addresses, alias comparison starts at 0<0,
   QROM writes other fields, and address arithmetic preserves zero addresses.
3. The first reversed operation in the earlier stage computes the final marker.
   At mu=nu=0, its controls (nu<=31, mu<=nu, aux4) are all one, because m1=0.
   It flips succ to one. Every remaining inverse gate preserves succ. Therefore
   PREPARE†|zero> has zero overlap with |zero>, making the reflection identity
   on the initialized record for either control value.
4. SELECT and reflection each preserve this subspace. Independent finite traces
   of all sixteen clock words establish restoration of clock and private work.
   Linearity then covers arbitrary coherent clock amplitudes.

This does not claim the full walk is identity for nonzero ELL, nor implement a
general tensor-hypercontraction chemistry solver. The original closed importer
checks its source structure separately; the native example does not import that
proof, Qmod source or test code. Tests independently simulate the twelve-bit
inverse early stage, check later-stage guard preservation, trace the clock and
compare exact amplitudes. Changing source marker or dirtying work remains
refused by importer regression tests.

Independent Circuit uses Householder clock preparation and an analytic inverse
DFT on the six active bits. Native and original keep 83 logical bits and execute
exact CPU MPS with maximum bond four. Validation checks all 64 occupied-support
complex amplitudes, unit total support probability (so every omitted inactive
word has zero probability), zero compiler workspace and no truncation. Dense
2**83 conversion is explicitly forbidden. Both default circuits have 43
operations; native and source depth are measured in the delivery evidence.

The complete Notebook's molecular data preparation and analysis remain pending.
