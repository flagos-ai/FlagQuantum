# Reversible elliptic-curve addition

The reference is the immutable Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. The batch covers
`ec_add.py`, its tests and tutorial, and their direct arithmetic dependencies.

## Stage one

The operation reads four big-endian Montgomery residues `a, b, x, y` and
returns them unchanged. It owns five zero-initialized outputs in source signature
order: `f1, f2, f3, f4, ctrl`. The predicates are `a == x`, `b == -y mod p`,
the two zero-coordinate pairs, and all-zero `(f2, f3, f4)`. The temporary negation
and restoration of `y` follows the source order. The caller separately provides
`2n` zeroed scratch qubits and `2n - 1` reusable measurement slots. The
source's all-ones modulus still fails the `p+1` constant load.

The original fast classical and symbolic-cost selection (33 tests) and the
unchanged 11-cell notebook have already executed. The remaining five stages,
the combined resource trace and executable native tutorial are open. The full
source-file rows remain open until the composed operation is verified.

### Stage one evidence

- Bound all seven operation bodies from their locked Soquet connections, including
  the negation/restoration of `y` and both all-zero coordinate checks. Compared
  wire order, gate operands and feedback slots at widths 2, 3, and 5. The
  reference graph permits independent operations to move; the native circuit
  follows source construction order while reusing one workspace.
- At width 3 and modulus 3, all 81 residue quadruples agree with the source
  callback. The per-AND certificate proves equal normalized complex output on
  every erasure branch and zero scratch. Three complete sparse executions on
  reversed wire bindings reproduce the full complex state.
- At width 3 and modulus 5, the emitted stream has 225 CX, 136 X, 84 H,
  56 T, 56 T†, 28 S, 28 conditional CZ, 28 measurements and 28 resets.
  The source model reports 28 AND, 213 Clifford and 28 measurements, with
  a 21-qubit modeled peak. The fixed native layout reserves 23 wires,
  including the five result flags and six scratch wires. The model can
  schedule zero-state allocations at their first use.
- The focused dependency regression passed 1,248 tests. Strict typing,
  formatting, team scope and architecture, IR and public API checks passed.

## Stage two

The native slope stage consumes the two through flags and five big-endian
Montgomery words, and writes the fresh `lam` output. It follows the source
subtractions, Kaliski inverse, dirty windowed multiply, and the two slope
selection loops. The equality predicate stays live through slope selection and
the negative-control update of `f1`. The adjoints consume the original inverse
history and multiplication index/reduction outputs. The original `x` word is
restored to its physical port; integration tests check zero on every work wire.

At width 3 and modulus 3, composing the flag and slope stages on reversed
physical wire bindings matches the source's first two stages for all 243
coordinate/tangent combinations. The per-AND certificate verifies all 498
erasures on this domain; two complete sparse executions compare complex states
with phase-sensitive inputs. The negative-controlled equality was compared
independently for primitive order on each wire, operands and feedback. The
existing component regressions cover the two subtractions, both inverse and
both windowed multiplication directions.

| Width, modulus, window | Native emitted operations | Source model |
|---|---|---|
| 3, 3, 1 | 3,056 T/T†, 6 CCX, 470 measurements, 71 bound wires | 470 AND, 6 Toffoli, 168 CSwap, 3,175 Clifford, 470 measurements, 55 peak qubits |
| 5, 13, 5 | 8,296 T/T†, 10 CCX, 1,194 measurements, 131 bound wires | 1,166 AND, 26 Toffoli, 480 CSwap, 8,407 Clifford, 80 T, 1,166 measurements, 97 peak qubits |

The measured adjoint streams and static workspace reservation explain why
primitive counts and modeled counts need their own columns. The first row has
5,920 CX, 1,746 H, 1,062 X, 470 S, 470 conditional CZ and 470 resets in addition
to the listed T/T†, CCX and measurements.

Two observable source boundaries remain:

- For `n=3, p=3, window=1, f1=ctrl=a=b=lam_r=0, x=y=1`, the high-level callback
  reports `f1=0`; the original one-level decomposition and the native circuit
  both produce `f1=1`, with zero slope.
- For `n=5, p=13, window=5, f1=a=b=lam_r=0, ctrl=1, x=2, y=9`, the source
  Kaliski inverse supplies 5 to the multiplier. Its decomposed multiply on
  `(5,9)` reaches 27 via the locked +p final reduction, and the circuit selects
  27. The stage callback and the one-level graph invoking the high-level child
  method both report 1. The native stage preserves the decomposed behavior.
  This leaves full mathematical curve-addition parity open.

## Stage three

The controlled coordinate update computes the dirty Montgomery product
`x * lam`, subtracts it from `y` under `ctrl`, and consumes the product and
its retained indices and reduction bit in the adjoint. The same layout
helper now serves stages two and three. A second chain copies `a`, doubles
the copy, adds `a`, and conditionally adds the resulting `3a` to `x`.
Add-adjoint and double-adjoint clear the temporary before the final XOR.
The double result uses the source's rotated word. Its former most
significant bit is in the returned work tuple until the adjoint restores
the original ordering.

At width 3, modulus 3 and window 1, the complete gate dependency order,
operands, including the rotated word, and measurement feedback agree with
the locked component bodies. All 243 initial residue/tangent combinations
through the first three stages match the source composition. The channel
certificate checks all 634 measurement erasures and zero work; two sparse
executions check full complex amplitudes on permuted physical wires. At width
4 and modulus 5, 250 inputs exercise both control values and nonzero tripling
with the source slope relation. The zero-control branch preserves both
coordinates and the remaining inputs.

| Width, modulus, window | Native emitted operations | Source model |
|---|---|---|
| 3, 3, 1 | 544 T/T†, 1,454 CX, 408 H, 152 X, 136 S, 136 conditional CZ, 136 measurements and 136 resets; 40 bound wires | 136 AND, 910 Clifford, 136 measurements, 35 peak qubits |
| 4, 5, 1 | 824 T/T†, 206 measurements, 50 bound wires | 206 AND, 1,334 Clifford, 206 measurements, 44 peak qubits |
| 5, 13, 5 | 1,408 T/T†, 312 measurements, 83 bound wires | 284 AND, 16 Toffoli, 80 T, 2,348 Clifford, 284 measurements, 76 peak qubits |

For width 5, modulus 13 and window 5, `ctrl=1, a=b=0, x=5, y=1, lam=9`,
the locked multiplication's decomposed result is 27 with reduction flag 1.
This supplies an out-of-residue operand to the controlled subtraction.
The primitive execution returns `y=6`, while the source stage callback
returns `y=0`. The measured native branch preserves the amplitude and returns
all temporary wires to zero. This is a counterexample for the executable
arithmetic chain outside the multiplication-to-subtraction residue promise.

Stages four through six, the combined source graph and the corresponding
native tutorial remain open.
