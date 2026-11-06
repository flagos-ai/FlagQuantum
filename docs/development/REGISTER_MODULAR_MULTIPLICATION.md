# Exact controlled exchange and modular multiplication

Run `python -m examples.algorithms.register_modular_multiplication`.
Users append `ModularMultiplication(value)` through the existing circuit append
protocol. The example checks coherent controls, complex residue amplitudes,
thirteen clean work wires and restoration by the inverse multiplier, without
NumPy or Qualtran installed. Compiler modules only emit native IR.

## Interfaces and original algorithms

`controlled_swap_bit_ir(control,left,right)` retains every primitive in pinned
TwoBitCSwap.to_clifford_t_circuit: seven T/T-dagger, eight CX and two H, no work
or measurements. `controlled_swap_ir(control,left_tuple,right_tuple)` applies
that exact circuit in the source CSwap bit order. Registers are disjoint,
equal-width, nonempty and can contain arbitrary entangled states. This is an
exact exchange, not a relative-phase approximate swap.

`controlled_modular_multiply_constant_ir(control,target,value,modulus,workspace,
classical_bits)` retains CModMulK's original sequence: allocate zero result y;
CtrlScaleModAdd(k,x,y); CtrlScaleModAdd(-k_inverse,y,x); CSwap(control,x,y);
free y. The final exchange must be controlled: unconditional output renaming
would destroy the inactive branch. All words are big-endian unsigned or
Montgomery bit representations, without an extra Montgomery scaling factor.
Input promises hold on both coherent control branches: 0<=x<p, 1<p<2**n,
0<k<p and gcd(k,p)=1. The compiler validates concrete parameters/layout, not
quantum amplitudes. Work starts/ends zero. The first n work wires hold result y;
remaining work reuses modular_scaled_requirements(n). Classical slots are
reused after inner cleanup and retain only latest outcomes.

`modular_multiplication_requirements(n)` returns n+scaled_work quantum wires
and the unchanged scaled classical budget. At n=8 these are 35 work and 16
classical slots; with the control and target there are 44 total qubits.
No runtime/backend, allocation policy, root public API or upstream source changes
are introduced. Apache-2.0 copyright/source links are retained.

## Source comparison and complex channels

Reference is unmodified Qualtran v0.7.0 commit
bcb17bb25360ed68ce3bc084645db3e929799389, basic_gates/swap.py and
mod_arithmetic/mod_multiplication.py. The reference adapter invokes the source's
explicit TwoBitCSwap.to_clifford_t_circuit because its atomic Bloq has no generic
decomposition. Other source stages preserve their component lifetimes and
original Bloq decomposition. Operands, feedback, gate multisets and each wire's
ordered stream agree; disjoint source gates may commute.

All eight columns of the literal three-qubit Fredkin circuit are independently
checked against the unmodified source unitary and the ideal positive-phase
permutation. All complex register inputs through width three, with an entangled
spectator, and involution are also checked. A test-only channel certificate
recognizes only this complete literal swap block, independently verified And
blocks, and unconditional X/CX. An incomplete block is rejected. It never
replaces production execution.

Multiplication tests cover both source dtypes, identity/nonidentity multipliers,
prime/composite moduli, widths 2/3/5/8. Every valid input basis state is checked
for all measurement branches using strict block invariants and linearity,
including zero work and restoration by inverse k. At n=8,p=251 all 502 coherent
control/residue basis inputs are covered. Full raw primitive projection at n=2
cross-checks zero/one/alternating measurement branches with nonuniform complex
phases and an entangled spectator. This is not an enumeration of 2**672 dense
trajectories or a dense 44-qubit simulation.

## Measured primitives versus model weights

| n=8 circuit | CX | X | H | T incl. dagger | Measurements | Work | Total qubits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CSwap, arbitrary words | 64 | 0 | 16 | 56 | 0 | 0 | 17 |
| CModMulK, p=251,k=173 | 7824 | 840 | 2032 | 2744 | 672 | 35 | 44 |

Multiplication also emits S=CZ=reset=672; conditional CZ execution depends on
outcomes. Measurements and QubitCount independently match the source models.
The unmodified resource model stores cswap=8, and_bloq=672, clifford=5136.
Its default total_t_count() weights each CSwap as **4 T**, giving **2720 T**.
The actual upstream exact circuit weights each as **7 T**: calling the existing
model API total_t_count(ts_per_cswap=7) gives **2744 T**, exactly the emitted
source circuit. Both values are asserted and disclosed. The 24-T difference is
8*(7-4), not a port-induced overhead. The model Clifford field is not the
expanded primitive Clifford total.

General exact primitive T cost is 40*n*n+23*n; the default source estimate is
40*n*n+20*n. Measurements are 2*n*(5*n+2). No default estimate is relabeled as
an executed count or silently substituted for literal source gates.

## Explicit boundaries

The source classical model claims identity for out-of-range x, but its layers
can disagree. For n=2,p=3,k=2,c=1,x=3: direct model gives x=3, one-level source
decomposition gives x=0, and fully flattened source raises on a nonzero Free.
The test retains all three observations. Native promises x<p in both branches;
it does not silently reset leaked work or correct the reference. Invalid/nonunit
multipliers are rejected, matching source constructor restrictions. Symbolic,
empty-word and other swap/modular families remain outside this concrete slice;
the parent upstream files remain in progress.
