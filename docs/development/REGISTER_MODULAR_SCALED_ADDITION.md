# Constant and scaled modular accumulation

The compiler register modules `modular_constant_addition.py` and
`modular_scaled_addition.py` extend the shared CModAdd path. They emit native IR
using existing constant XOR, CModAdd, And, and measurement-assisted And adjoint.
No new executor, runtime policy, device selection, or public root API is added.
Run the coherent example with:

```bash
python -m examples.algorithms.register_modular_scaled_addition
```

Users write `circuit.append(ModularAccumulation(operation, value))`. The example
checks both coherent control branches, all two-bit source values and valid
target residues, complex amplitudes, work cleanup and inverse-value restoration
using the existing dynamic runtime. The constant case also keeps an entangled
source register as spectator. It runs without Qualtran or NumPy installed.

## Interfaces and ownership

`controlled_modular_add_constant_ir(control, target, value, modulus, workspace,
classical_bits)` implements the pinned positive-control CModAddK stages:
IntState(k), CModAdd, IntEffect(k). Work is initially/ultimately zero; its first
n wires hold the literal big-endian constant while CModAdd uses the rest.
`modular_constant_requirements(n)` gives `(n+max(2*n+2,n+6), max(2*n,n+2))`.
Both control branches promise `0 <= target < modulus`, with
`0 < modulus < 2**n` and `0 <= value <= modulus`. The value is not normalized
modulo p: that would change the source's executed algorithm for invalid inputs.

`controlled_modular_scaled_add_ir(control, left, right, scale, modulus, workspace,
classical_bits)` preserves control/left and updates right to
`(right + scale*left) % modulus` on the positive branch. Left may be any n-bit
unsigned word, right must lie in `[0,modulus)` on both branches, and concrete
scale may be negative. `modular_scaled_requirements(n)` adds one quantum work
wire to the constant-add budget. All operand/work/control wires are disjoint.
Classical slots are reused after inner cleanup and retain latest outcomes.

The scaled path follows the original high-to-low traversal exactly. For each
bit i it computes And(control,left[i]), calls CModAddK with
`(scale * 2**(n-1-i)) % modulus`, then erases that And. Zero constants still emit
the full source stage, including when scale is zero. Source control dependencies
serialize the lifetimes, so the same And target, loaded constant bank and inner
work pool are reused. The outer erase uses an existing classical slot only after
the inner component has finished reading it. The compiler never inspects input
amplitudes to validate a residue promise.

## Source parity and preparation/effect boundary

Both modules retain Apache-2.0 Google copyright and pinned source attribution.
Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`,
`mod_arithmetic/mod_addition.py::CModAddK` and `CtrlScaleModAdd`, is unmodified.
Tests retain parent call lifetimes and compare original primitive operands,
per-wire ordered streams, gate multisets and classical feedback. Disjoint gates
in the source DAG may commute. Inner arithmetic uses original upstream Bloq
controlled decompositions, rather than generic Cirq controlled synthesis.

IntState/IntEffect are states/effects, not unitary gates; upstream ZeroEffect has
no Cirq export. The test adapter checks the unmodified decomposed source tensors
against the exact positive basis vector/covector and binds them to initially
zero native work using X load/unload. Whole-channel proofs independently show
the returned work is zero, making the source effect succeed with unit amplitude
on the promised domain. This explicit representation boundary is not a claim
that upstream's unsupported Cirq export succeeded, and no source is patched.
The comparison of actual arithmetic gates is otherwise unchanged.

## Complex channels and all measurement branches

The shared test-only exact And channel certificate checks the literal primitive
blocks and every compute/erase invariant on every valid basis input. Separate
three-qubit primitive checks establish both Kraus outcomes, so composition and
linearity prove all `2**m` complete branches have identical output amplitudes
multiplied by `2**(-m/2)` and zero work. The certificate rejects unknown gates;
it does not replace production execution. Raw full-IR projections cross-check
complex amplitudes with an entangled spectator, every one-bit-constant branch,
and zero/one/alternating branches for wider cases. Inverse scale is verified
on all valid basis inputs through width eight, including negative/zero scale.
For n=8,p=251 this includes all 128,512 control/source/target basis inputs.

## Measured primitives versus independently evaluated estimates

The table scans emitted IR at n=8, p=251, k=173. T includes T-dagger;
conditional CZ is an emitted instruction and executes only on its selected
outcome. One reset per measurement. H=3m, S=CZ=reset=m for m measurements.

| Native emitted circuit | CX | X | T | Measurements | Clean quantum work | Total qubits | Reused classical slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CModAddK | 479 | 55 | 164 | 41 | 26 | 35 | 16 |
| CtrlScaleModAdd | 3880 | 422 | 1344 | 336 | 27 | 44 | 16 |

Independently evaluated upstream QECGatesCost gives CModAddK `20*n+4` T and
`5*n+1` measurements; CtrlScaleModAdd gives `n*(20*n+8)` T and `n*(5*n+2)`
measurements. QubitCount gives 35 / 44 for these instances. These agree with
native primitive T/measurement scans and explicit total qubit budgets.
The source model separately contains `and_bloq=41, clifford=320` for CModAddK,
and `and_bloq=336, clifford=2568` for scaling. Those Clifford fields do not count
the expanded And primitive Clifford instructions and model basis loads as
ArbitraryClifford. They are **not** measured primitive Clifford totals, and are
not compared directly to CX/X/H/S/CZ scans. Full measured counts are printed by
the example and checked independently in tests.

## Open source boundaries

Upstream CModAddK's classical model accepts fitting k outside `[0,p]`, but its
executed decomposition can disagree: n=2,p=2,k=3,control=1,target=1 gives source
classical target=0 while the unmodified primitive path gives target=2. The test
retains this counterexample; native rejects that domain, without normalizing or
silently correcting the reference. A constant that does not fit n bits also
fails upstream IntState construction. Symbolic inputs, the separate ModAddK
class (which has no gate decomposition in this source), and other unfinished
modular families remain open. G2 is still in progress.
