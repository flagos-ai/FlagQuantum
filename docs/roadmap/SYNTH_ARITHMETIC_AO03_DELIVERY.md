# AO-03 bounded Boolean value candidate

Revision 9 §11.8 introduces a compiler-internal candidate for small exact
unsigned expressions. It uses the verified expression and its normalized SSA
and width evidence. The existing public synthesis path still uses its existing
selector. AO-04 owns target-aware candidate selection.

Each output bit is an algebraic normal form (ANF), with a set of square-free
monomials. XOR cancels duplicate monomials, and multiplication unions variable
sets because a Boolean variable squared equals itself. Ripple carry uses
`ab XOR ((a XOR b) AND carry)`, the Boolean identity for the majority of its
three inputs. Multiplication sums shifted bit products. Shared SSA values are
expanded once and read by their consumers. These identities apply across
operator boundaries, including multiplication followed by addition.

The mapper writes every term into the existing XOR output using X, CX or CCX.
A cubic term computes two controls into one zero workspace bit, applies a
Toffoli with the third control, and reverses the first Toffoli. Input wires are
read-only, arbitrary output contents are preserved under XOR, and the same
workspace can serve later terms. The candidate records the output ANFs, their
input-wire mapping, the returned SSA value and used workspace.

The fixed ceilings are six input bits, twelve bits per output or expanded
intermediate, 64 inlined source operations (including return), 20,000 term
products, and at most 256 mapped gates or the caller's lower operation limit.
At six input bits a polynomial contains at most 64 distinct monomials. The
implementation expands algebraically without generating a whole-domain truth
table. Degree greater than three or unavailable clean scratch yields a mapping
refusal. Unsupported operations and exhausted budgets yield distinct refusals.
Invalid bindings still use the existing contract diagnostics. The candidate
does not claim coverage or global optimality beyond these limits.

## Logical resource comparison

The following figures include a Hadamard on each input wire. Width counts
input and output wires plus workspace actually used. A comparison harness
can give the existing scheduler a larger workspace pool.

| Expression and input widths | Existing width / scratch / gates / depth | ANF width / scratch / gates / depth |
| --- | --- | --- |
| `x*x+y`, 2 and 2 | 16 / 8 / 81 / 59 | 9 / 1 / 14 / 9 |
| `x**2`, 2 | 14 / 8 / 64 / 49 | 6 / 0 / 6 / 4 |
| `3*x+1`, 2 | 13 / 7 / 128 / 119 | 6 / 0 / 8 / 5 |
| `x*y+1`, 2 and 1 | 10 / 4 / 32 / 25 | 7 / 1 / 10 / 7 |

The sentinel candidate fits the 9-wire, 1-workspace, 24-operation and depth-20
ceilings. Its dense complex128 state buffer is `2**9 * 16 = 8192` bytes.
These are logical candidate measurements. The public qfunc and Qmod paths
retain the existing resource result until selection is integrated.

See the [existing qfunc, Qmod and hand Circuit comparison](SYNTH_PYTHON_PERFORMANCE_PF01_DELIVERY.md#existing-user-workflow).
The new compiler tests check complete domains and arbitrary XOR outputs for
seven expression/width configurations, strict operation improvements against
the existing scheduler, 256 sentinel input/target pairs, complete complex128
amplitudes on the sentinel's zero-workspace subspace, and budget versus
mapping refusal. No frozen baseline or oracle was modified.
