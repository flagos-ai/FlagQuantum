# Signed velocity and coherent finite differences

`main.py` prepares seven data bits entangled with seven reference bits and eight
block bits. `operations.py` composes periodic increments, derivative boundary
conditions, signed velocity predicates, velocity coupling and LCU preparation.
Users specify numeric register views and coherent controls; synthesis assigns
physical storage and cleans temporary work.

Slices use least-significant-bit indices. The combined word `[x, flag]` places
three position bits below the flag, so increment carries from x into flag.
Velocity uses signed three-bit two's-complement values -4 through 3. Whole
signed words retain that meaning; raw unsigned slices use their selected bit
order. Borrowing remains conservative when a predicate and its target share a
parent register.

The state axes are electric, x, velocity, reference_e, reference_x, reference_v,
selector, second, first, indicator, block, zeta and flag, followed by two clean
work bits. The seven Bell reference pairs exercise all 128 data basis inputs
coherently. No postselection or phase alignment is used in acceptance.

`comparisons/circuit.py` constructs local matrices directly from mathematical
permutations, prefix rotations and the frozen coefficients. All 4194304 logical
complex amplitudes must agree at absolute tolerance 1e-11; aggregate dirty-work
probability must remain below 1e-22. A separate test checks this source-free
reference against the source-derived independent mathematical reference.

The native result uses 24 physical bits, including two clean work bits, 22976
operations and depth18578. Budgets remain 24 physical bits, two work bits,
100000 operations and 256 MiB planned-state allocation. Process RSS, which also
includes Python, numerical libraries and independent reference buffers, is a
separate measurement. The frozen original uses 4467 operations and depth3359.
Exact scratch-free phase decomposition trades gates for work; this establishes
semantic and budget acceptance, without a resource-parity claim.

Run the formal acceptance from the repository root:

```sh
python -m pytest tests/integration/test_synthesis_native_vlasov.py -q
```
