# Native molecule QPE with exact power-specific Strang schedules

Status: accepted locally. 22 new molecular QPE tests and 23 time-marching
regressions pass, with 29 catalog/API tests and strict typing of five files.
Final fresh 216-original hashes, outcomes and resources exactly match prior;
approved six pass and all 29 production pins are unchanged. Native coverage
is 204/216; Notebook workflows remain 24/213.

Native default: 13 physical bits, zero work bits, 17,957 operations and depth
16,199. Original: 13 bits, zero work bits, 16,931 operations and depth 15,488.
Default complete amplitude errors against independent ungrouped Circuit are
below 2e-14. Changed signed input/scale and coherent power/inverse trials pass;
the latter explicitly verify their one preparation scratch bit returns to zero.

Retain the complete frozen LiH model: all 49 ordered Pauli terms including the
identity coefficient, six zero-initialized state bits, seven phase bits and
seven controlled evolution powers. Power p=2**i uses time=-2*pi*p and
ceil(0.05*p**1.5) second-order repetitions, namely [1,1,1,2,4,10,26]. Do not
replace these by powers of a fixed approximate one-step unitary: product-formula
approximations do not obey that identity. Preserve signed phase interpretation
word/128 with two's-complement decoding and the inverse-QFT bit convention.

Contiguous mutually commuting term runs have sizes [10,4,4,6,6,1,7,11]. Combining
the central run's two half steps and the boundary run's adjacent half steps is
an exact regrouping of the declared Strang sequence. Retain all coefficients
and order between noncommuting runs; independently verify commutation and model
identity. This yields a bounded native composition without raising the existing
4096-effect cap. No claim of a new optimizer or general automatic grouping API.

Reuse public Observables, bound real/int arrays, suzuki_trotter, scalar coherent
control, raw array views and inverse qft. A raw seven-bit output preserves the
source's signed fractional encoding with explicit numeric decoding in teaching
prose. No production Compiler, API, ownership rule or budget change is planned.

Independent Circuit acceptance must construct ungrouped Pauli exponential
rotations from the original order and each distinct step count, with full
identity-controlled phase and exact inverse QFT. Compare all 8192 complex
amplitudes against native and frozen source without discarding global phase.
Exercise an additional nonzero input state and changed overall time scale,
individual controlled powers on coherent nonzero controls, inverse restoration,
model/group/schedule identity, alias rejection and resource budgets. Preserve
original resource baseline and report any native overhead.

After final acceptance run a fresh 216-original plus approved-six replay,
archive immutable evidence and update catalog/ledger. Notebook workflows remain
separate at 24/213. Classification is `single_device_fast_path`. Do not push,
open the integration PR or run full remote CI before all 216 native quantum
programs have independent acceptance.
