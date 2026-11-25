# Complete Hidden Shift workflow

The frozen four-bit quantum counterpart in the parent directory is unchanged.
This separate workflow migrates all thirteen original Notebook code cells:

| Source cells | Native workflow |
| --- | --- |
| 4 | Existing optional SymPy replaces installation of galois; no runtime pip |
| 6,7 | Four-bit shifted bent function, native synthesis, resources and 2048 shots |
| 10,11 | Seed-1 permutation and inverse; original sixteen-bit function and shift |
| 13,14,15 | Complex dual route, full state, 2048 observations and source shift assertion |
| 19,20 | No-dual indicator algorithm and 2048 joint observations |
| 22,24,26 | Select independent binary equations, solve over GF(2), assert recovered shift |

The source's shift helper uses a threshold of `NUM_VARIABLES >> 2`, rather than
the half-register boundary. The supplied shifts 1,3,9 select the same intended
locations under that expression. This workflow preserves precisely that supplied
case; it does not claim that the source helper works for arbitrary shifts.
The simple source shifts bit zero, unlike the frozen quantum counterpart.

XOR of Boolean terms becomes a phase sum modulo 2*pi in the dual algorithms.
The no-dual marker uses direct checked Boolean `^=` on static array views,
including the eight-way AND. Compiler reuses bounded predicate expansion and
indexed controlled-X effects; no new physical opcode or whole-register numeric
scratch is introduced. Ordinary integer arithmetic retains its existing path.
The admission is one writable QBit, disjoint one-bit reads, constants 0/1 and
&, |, ^, ~, with at most eight distinct input bits per update. Ownership,
initialization, static indexing, effect limits and physical budgets remain checked.

Independent Circuit oracles use pairwise CZ/Toffoli gates plus bounded eight- or
nine-bit matrices. A dense sixteen-/seventeen-bit oracle is never constructed.
Every absolute amplitude is compared at 2e-11, without global-phase alignment,
and every clean-work amplitude is checked at squared norm below 1e-24. The no-dual
indicator uses exact Z semantics, including the constant phase convention.
Independent Walsh spectra verify all 65536 values of the bent-function dual.

Each route samples the actual named distribution with the original 2048 shots.
Seeds 1500/1501/1502 make local runs reproducible; the hosted source was unseeded.
The same seeds applied to independent Circuit probabilities must give identical
observations. Binary rank selection and inversion use SymPy over GF(2), including
a regression where real rank would give the wrong answer. Insufficient,
nonbinary and inconsistent equations are rejected.

```sh
python -m pytest tests/integration/test_synthesis_hidden_shift_notebook.py tests/integration/test_synthesis_native_boolean_xor.py -q
```
