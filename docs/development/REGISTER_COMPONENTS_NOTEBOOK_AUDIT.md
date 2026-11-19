# Complete component properties tutorial

Q-0944 binds all 38 cells of the unchanged frozen
`tutorials/quantum-programming/2-bloqs.ipynb`, Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`8ab4e2b48c82b0104d6b89340128d21386a5fa8f`.
`examples/notebooks/register_components.ipynb` maps every cell in order and adds
one provenance and scope cell, retaining the Apache-2.0 source attribution.

## Ownership and interface

Core owns the signed MSB-first through-register signature. Notebook-local frozen
`NegationDeclaration` and `Negation` values illustrate declaration followed by
implementation. The implemented value delegates physical emission to existing
Compiler negation, classical execution to Simulation, dense tensor enumeration
to the existing classical action tensor supplier, recursive QEC costs and peak
width to existing Compiler queries, and visualization to Drawer. Arithmetic
resource declarations supply the complete child dependency recipes explicitly.
No production API, algorithm, numerical kernel or dependency is introduced.

The native entry is `fq.Circuit(23).append(Negation())`. Concrete emission binds
n target wires, n clean constant-load wires, n−1 clean carries and n−1 classical
slots in fixed consecutive order. The caller owns these resources. A larger
circuit can retain idle spectator wires. Large and symbolic examples inspect
hierarchy without emitting gates. Symbolic physical emission fails explicitly.

## Whole source coverage

| Original cells | Native workflow |
| --- | --- |
| 0–2 | Recreate the eight-bit complement and increment primitive composition |
| 3–10 | Declare and instantiate a signed component, inspect signature and its signature-only width |
| 11–18 | Implement and append the reusable component, draw its opaque box and connected two-child decomposition |
| 19–24 | Explain concrete flattening, query recursive QEC cost and peak width, draw declared calls with queried costs |
| 25–28 | Evaluate x=5 as −5 and construct the complete 256×256 phase-free tensor |
| 29–33 | Instantiate a parameterized family and inspect the 2048-bit opaque and direct-call hierarchy |
| 34–37 | Preserve symbolic signed widths, both arithmetic children and symbolic resource expressions |

All original code cells execute unchanged. Every native code cell executes with
Qualtran, Cirq and Kickmix imports rejected. Native diagrams retain component
boundaries and direct calls through existing Drawer representations. Graphviz
and IPython are optional presentation dependencies. Dataclasses and explicit
resource binding replace the original abstract base class, attrs and implicit
builder allocation. The declaration and implementation are separate named
Python classes to keep extracted notebook code statically checkable.

## Acceptance evidence

The complete eight-bit tensor equals the original tensor entry for entry.
All 256 signed classical values agree, including fixed-width minimum wrap.
The eight-bit emitted stream matches original gate multiplicities, per-wire
dependency order, operands and measurement feedback. Actual `circuit.append`
execution is separately checked with a three-bit coherent word entangled with
a spectator. All four measurement histories match every complex unnormalized
amplitude with factor 1/2, with exact zero support on all five work wires.
Eight-bit full quantum measurement-history enumeration is outside this scope.

Concrete widths 2, 3 and 8, plus symbolic width n, match original QEC categories
and peak-width queries. Width 2048 matches the original signature, direct
children, drawing workflow and QEC categories. An extra peak-width query on the
unmodified original raises RuntimeError from recursion depth exhaustion. The
native peak is 6143. This additional query is a documented difference, with an
explicit regression assertion, and is outside the original tutorial's large
example, which only draws the hierarchy. The existing one-bit signed AddK(1)
constant-load defect remains outside this tutorial's supported concrete examples.

| Category | Value |
| --- | --- |
| Eight-bit actual primitives | 168, comprising X10, H21, T14, T-adjoint14, CX81, S7, measure7, conditional CZ7, reset7 |
| Independent complete negation model | 7 AND, 56 Clifford, 7 measurements |
| Independent peak model | 23 logical qubits at width8 |
| Physical ownership | 8 logical target, 8 constant-load work, 7 carry work, 7 classical slots |
| Exhaustive quantum history check | 3 target, 5 work, 1 spectator, all four histories |

The complete model includes the eight complement Clifford gates. Its remaining
48 Clifford gates belong to AddK. Model categories and emitted primitives use
different decomposition levels and remain separately reported.

Concentrated regression passes 121 cases with zero skips and 14 retained source
and optional-display warnings. Strict mypy covers the new test and every
extracted native code cell. Black checks the Python test and individually formats
notebook cells because the optional Jupyter formatter dependency is absent.
Ruff checks both the notebook and test directly. Scope, architecture and all 74
IR/public API checks pass. The whole-repository dependency scan retains 706 existing findings and zero
findings on the new paths. The whole dependency policy remains failing.

Persistent evidence uses `ftqc-reference-env/g2-components-*`. The initial probe
caught a Signature indexing error, which was corrected, and the extra 2048-bit
source recursion boundary, which remains asserted. The independent dashboard
numerical boundary remains open in `PHYSICAL_RESOURCE_DASHBOARD_AUDIT.md`.
