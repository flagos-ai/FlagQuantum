# Source-exact modular doubling and its adjoint

Run `python -m examples.algorithms.register_modular_doubling` without Qualtran
or NumPy. Users append `ModularDoubling(target, workspace, inverse)` and pass
`op.outputs` to the next operation. The example verifies both directions,
nonuniform complex residue amplitudes, clean returned work and complete
restoration of the original physical state and port roles.

## Interface and original output ownership

`modular_double_ir(target, modulus, workspace, classical_bits)` emits pinned
ModDbl. `modular_half_ir` emits its exact source adjoint. Inputs are concrete
big-endian unsigned or Montgomery bit representations, with 0<=x<p, a positive
odd p<2**n. No additional Montgomery scaling occurs. Both require 2n+5 zero
work wires and n+1 distinct reusable classical slots. Work roles are lower,
sign, n+2 constant wires, n+1 carries. The two arithmetic stages reuse their
constant/carry pool after cleanup; slots record latest outcomes.

The source does not return its result on the original physical target tuple.
It frees the old input MSB and returns target[1:] followed by the allocated
lower wire. `modular_double_outputs(target,workspace)` explicitly returns that
result tuple and clean work tuple `(old_target_MSB,sign,constant...,carry...)`.
The inverse `modular_half_outputs` returns `(old_work_lower,*target[:-1])` and
clean work `(old_target_LSB,sign,constant...,carry...)`. Applying the opposite
binding restores both original tuples exactly. Initially allocated lower can
become data; the *returned* work is zero. No hidden physical SWAP or claim that
the old work tuple stays zero is made. This is a compiler-level port binding,
not a new runtime, allocator, device policy or root public API.

Forward preserves signed AddK(-p) on `(sign,*target,lower)`, sign-controlled
AddK(p) on `(*target,lower)`, and X(lower)/CX(lower,sign)/X(lower). Equal-width
signed constant bit loading uses the existing exact shared arithmetic network.
The adjoint reverses components and calls exact controlled AddK and AddK
adjoints with the original constants. It does not use AddK(-k) as a substitute
or reverse a measured primitive stream. It produces inverse-two multiplication
`(y+p*(y&1))//2` on valid residues.

## Pinned source and validation

Apache-2.0 Google source attribution is retained. Unmodified Qualtran v0.7.0,
commit bcb17bb25360ed68ce3bc084645db3e929799389,
mod_arithmetic/mod_multiplication.py::ModDbl and its original Adjoint are the
references. Tests compare primitive multisets, each wire's order, operands,
feedback and actual returned source ports (independent gates may commute).
Temporary names are bound to the existing source allocation order; the source
is not patched to force fixed output wires.

Both QUInt and QMontgomeryUInt, widths 1/2/3/5/8, trivial/nontrivial odd moduli,
and both directions are covered. Every valid basis input satisfies the strict
And block invariants, proving all measurement branches have the ideal complex
map times 2**(-m/2), with returned work zero. This supports arbitrary coherent
input phases by linearity. Opposite-direction composition restores inputs and
ports. Raw full primitive postselection also checks complete complex inputs
entangled with a spectator: every 8/32 branch at widths one/two, and selected
zero/one/alternating branches at width three. This is not a claim to enumerate
all 2**17 dense eight-bit trajectories. The runtime example separately tests
append and sampled dynamic execution.

## Measured resources and independent estimates

At n=8,p=251, each direction emits 213 CX, 10 X, 68 T/T-dagger, 51 H, 17 S,
17 measurements, 17 conditional CZ and 17 resets. Conditional execution depends
on outcomes. There are 21 clean returned work wires, 29 total qubits and nine
reused classical slots. The source QECGatesCost independently gives 8n+4 T and
2n+1 measurements; QubitCount gives 3n+5. These agree with emitted counts and
explicit port ownership. Model Clifford fields are kept separate from the
expanded primitive Clifford inventory: the source reports and_bloq=17 and
clifford=138 at n=8,p=251, whereas expanded CX/X/H/S/CZ sum to 308.
No estimate replaces measured gates.
The direction-specific literal streams are compared, not merely their totals.

## Explicit source boundaries

Only the promised residue subspace is accepted. The source classical model is
identity for x>=p, but the actual decomposition need not be: n=2,p=1,x=2 gives
source classical x=2 versus decomposed/native x=3, even with clean returned work.
This counterexample is retained without correcting the reference. Native rejects
even, nonpositive and oversized moduli explicitly; symbolic inputs, full-domain
identity guarantees and generic port-routing policies remain outside this slice.
