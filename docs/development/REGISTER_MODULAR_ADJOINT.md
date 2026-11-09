# Original modular addition and subtraction adjoints

Run `python -m examples.algorithms.register_modular_arithmetic` without Qualtran
or NumPy. Users still write `circuit.append(ModularArithmetic(operation, inverse))`.
The example runs both directions of both original operations on coherent complex
residues, verifies all quantum work is zero, and restores the input with the
opposite direction of the same original operation. It does not substitute a
different add/subtract algorithm for an exact component adjoint.

## Compiler ownership and interfaces

`modular_add_adjoint_ir` belongs to compiler/registers/modular/modular_addition.py and
has the same signature as modular_add_ir: left, right, modulus, workspace,
classical_bits. Registers are big-endian, x lies in [0,p], y in [0,p),
1<=p<2**n. It preserves x and updates y=(y-x)%p, keeping original x/y ports.
The original _modular_workspace validation and layout own 2n+2 clean work wires
and n distinct classical slots. No new representation, runtime or device policy
is introduced; the protected root API is unchanged.

The inverse retains original component order:

1. X(sign).
2. Original unsigned LinearDepthGreaterThan adjoint.
3. sign-controlled AddK(p) adjoint.
4. extended-target AddK(-p) adjoint.
5. extended-word Add adjoint, with the original junk/sign zero prefixes.

The original constants are not normalized into another forward subtraction
network. Each shared component cleans its work before the next uses the pool.
Quantum work starts/ends zero; classical slots retain latest random outcomes.

`modular_subtract_adjoint_ir` belongs to modular_subtraction.py with the same
signature and budget as its forward counterpart. Both operands lie in [0,p),
and p+1 must fit n bits. Forward/inverse share validation and the original
conjugation shell: BitwiseNot, AddK(p+1), middle, AddK(p+1) adjoint, BitwiseNot.
Only the middle changes from ModAdd to its original exact adjoint. The inverse
therefore adds x modulo p, but it is a different source stream from ModAdd.
Similarly, ModAdd adjoint is different from ModSub. Tests retain their different
T/measurement counts instead of treating logical equivalence as gate equality.

## Independently verified comparator reuse

The unsigned LinearDepthGreaterThan component circuit is a palindrome around
copying out a sign: original compute/erase pairs exchange adjoints, and the
original copy/restore block is self-adjoint. Reuse of linear_greater_ir is
certified directly against the unmodified source adjoint at n=1/2/3/5/8.
Its inverse source allocations discover the b-sign before a-sign; reference
bindings follow that order without modifying source logic or native work.
The n=1 original MultiControlX special case remains intact.

Every comparator column with an arbitrary zero/one target is independently
checked at n=1/2/3. Full raw complex inputs entangled with a spectator and all
2/4/8 measurement histories verify target XOR, phase and zero work. This is
concrete unsigned comparator-adjoint acceptance, not signed/symbolic support or
a general inversion of measured primitive IR.

## Pinned source and complex channel evidence

The unchanged reference is Qualtran v0.7.0 commit
bcb17bb25360ed68ce3bc084645db3e929799389, mod_addition.py::ModAdd,
mod_subtraction.py::ModSub, arithmetic/comparison.py::LinearDepthGreaterThan
and their original adjoint objects. Google Apache-2.0 attribution is retained.
The source adapter keeps modular component lifetimes before decomposing their
original arithmetic/unsigned comparator components. Tests compare complete gate
multisets, ordered per-wire dependencies, operands and measurement feedback;
independent commuting instructions may reorder.

Both inverse circuits cover every valid basis input through n=8 via the strict
literal-And channel certificate, including x=p for ModAdd adjoint. Independently
verified And columns and both erase outcomes establish every full complex
measurement history with factor 2**(-m/2) and clean quantum work. Raw ModAdd
inverse channels enumerate all n=1/2 histories (8/128); ModSub inverse enumerates
all 512 histories at n=2. Larger n=3 channels use zero/one/alternating histories
with entangled spectator amplitudes. This is a compositional all-history proof,
not dense enumeration of 2**45 trajectories. Original forward restoration and
all preexisting forward consumers remain in the centralized regression.

The upstream y=p clean-domain exception, p+1 overflow for all-ones moduli,
symbolic parameters and further datatype/control families retain their separate
open ledger entries. The native clean-input promise is not silently widened.

## Measured primitives and separate model estimates

At n=8,p=251, T includes T-dagger. CZ below is an emitted conditional feedback
instruction; actual execution depends on outcomes. Each erase contributes one
reset. Quantum work and classical slots are reused across verified clean stages.

| Original inverse | CX | X | T | H | Measurements / S / conditional CZ / reset | Clean work | Total qubits | Classical slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ModAdd adjoint | 376 | 27 | 124 | 93 | 31 | 18 | 34 | 8 |
| ModSub adjoint | 538 | 67 | 180 | 135 | 45 | 18 | 34 | 8 |

Independent pinned QECGatesCost gives respectively 16n-4 and 24n-12 T,
4n-1 and 6n-3 measurements. QubitCount independently agrees with 4n+2 total
qubits. Its abstract and_bloq/clifford fields at n=8 are 31/248 and 45/380;
these are not primitive Clifford inventories. Equal forward/inverse totals do
not imply equal ordered streams or instruments with different classical histories.
