# Original modular multiplication and negation inverse chains

Run these examples using only FlagQuantum's existing native dynamic runtime:

```bash
python -m examples.algorithms.register_modular_multiplication
python -m examples.algorithms.register_modular_negation
```

Users still write `circuit.append(op)`. Both examples run forward and exact
source-adjoint directions, check full complex coherent amplitudes and clean
quantum work, and restore the input using the opposite original direction.
The negation example includes both control polarities and the ordinary path.
The small multiplication example uses k=2 modulo 3, which happens to be its own
inverse; separate source/channel tests include non-self-inverse multipliers.
No Qualtran or NumPy is required to run the examples.

## Ownership, interfaces and source order

All expansion belongs to compiler/registers. No runtime, backend, protected root
API or new IR representation is introduced. Big-endian logical ports and caller
work ownership are unchanged. Quantum work starts/ends zero; classical slots
retain latest random measurement outcomes and are not claimed to reset to zero.

| New compiler entry | Owning module | Clean input promise | Work / classical slots |
| --- | --- | --- | --- |
| controlled_modular_multiply_constant_adjoint_ir | modular_multiplication | Positive control; 0<=x<p, 0<k<p, gcd(k,p)=1 | n + modular_scaled_requirements(n)[0] / modular_scaled_requirements(n)[1] |
| modular_negate_adjoint_ir | modular_negation | 0<=x<p; 0<p<2**n-1 | 2n / n-1 |
| controlled_modular_negate_adjoint_ir | controlled_modular_negation | Both branches 0<=x<p; 0<p<2**n-1; control_value 0 or 1 | 2n+1 / max(1,n-1) |

Each entry has the same signature as its existing forward counterpart. Shared
private bodies retain validation and source components; direction changes only
the original component order and choice of exact component adjoint.

CModMulK's inverse allocates the same zero result word, executes original CSwap,
then CtrlScaleModAdd(-k^-1) adjoint from result to target, then
CtrlScaleModAdd(k) adjoint from target to result, and returns zero result work.
The pinned CSwap.adjoint() explicitly returns itself, so it reuses the original
exact seven-T CSwap stream. It is not implemented by reversing measurement
instructions or by synthesizing the forward multiplier with k^-1. The source
scale constants and their reverse bit traversals are retained.

ModNeg's inverse executes X(flag), original zero test, controlled AddK(p+1)
adjoint, original fanout, zero test, X(flag). CModNeg retains its original
And(control,flag) compute and measured erase lifetime around the exact
AddK(p+1) adjoint followed by fanout. Both gates are logical involutions on
residues, but neither inverse stream is identified with the forward stream.
Tests explicitly reject that shortcut.

The modular constant-adjoint source allocation binding is now shared between
constant, scaled and multiplier tests. It only renames original temporary
quantum/classical identifiers; original source logic is unchanged.

## Pinned reference and complex channels

The reference is unmodified Qualtran v0.7.0 commit
bcb17bb25360ed68ce3bc084645db3e929799389: mod_multiplication.py::CModMulK,
mod_subtraction.py::ModNeg/CModNeg and their original adjoint objects. Google
Apache-2.0 attribution is retained in the native owning modules. Tests read the
actual source components, compare gate multisets, ordered per-wire dependencies,
operands and feedback. Independent commuting gates may reorder.

All valid residues through n=8 are checked by the strict primitive And/exact
CSwap channel certificate. Original exact CSwap columns and both And erasure
branches were independently established by the existing shared tests. Linearity
and clean component composition prove full complex amplitudes, zero work and
branch factor 2**(-measurements/2) for every outcome history. All raw ModNeg
branches are also enumerated at n=2/3 (8/64 histories); CModNeg at both polarities
uses 16/128 histories, with an entangled spectator. Multiplication raw n=2
projections use zero, one and alternating histories plus the compositional
all-history certificate. It does not densely enumerate 2**672 trajectories.
Original-direction restoration and the append examples independently validate
consumer execution. Forward source streams stay in the centralized regression.

Inherited all-ones-modulus loading failures, out-of-residue dirty flag/multiplier
cleanup exceptions, symbolic inputs and generic controls remain outside this
concrete acceptance. They are not silently normalized or fixed in the reference.

## Measured primitive resources versus independent estimates

At n=8,p=251,k=173 (where relevant), T includes T-dagger. Counts below are emitted
IR instructions, including conditional CZ; actual feedback execution is
outcome-dependent. Every erasure emits one reset. Work and classical slots are
reused after clean component boundaries.

| Original inverse | cv | CX | X | T | H | Measurements / S / conditional CZ / reset | Clean work | Total qubits | Classical slots |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CModMulK adjoint | 1 | 7,824 | 840 | 2,744 | 2,032 | 672 | 35 | 44 | 16 |
| ModNeg adjoint | none | 194 | 66 | 84 | 63 | 21 | 16 | 24 | 7 |
| CModNeg adjoint | 1 | 200 | 66 | 88 | 66 | 22 | 17 | 26 | 7 |
| CModNeg adjoint | 0 | 200 | 70 | 88 | 66 | 22 | 17 | 26 | 7 |

Independent source QECGatesCost gives ModNeg 12(n-1) T / 3(n-1) measurements,
and CModNeg four T / one measurement more. Their abstract and_bloq/clifford
fields at positive control are 21/155 and 22/156. They are not primitive
Clifford inventories.

For CModMulK, the source default assigns each CSwap four T, so its default
estimate is 2,720 T. Evaluating the same source cost with ts_per_cswap=7 gives
2,744 T and matches the actual source seven-T decomposition. The difference is
8*(7-4), not a native synthesis penalty. Its other abstract fields are cswap=8,
and_bloq=672, clifford=5136, measurement=672; these remain separate from the
primitive scan. Source QubitCount independently matches all tested layouts.
Equal forward/inverse totals do not imply identical ordered streams or
measurement instruments.
