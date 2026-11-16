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

Stages three through six, the combined source graph and the corresponding
native tutorial remain open.
