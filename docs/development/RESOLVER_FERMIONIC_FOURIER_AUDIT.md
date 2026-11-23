# Complete fermionic Fourier constructor disposition

## Source and ownership

The complete qft/two_bit_ffft.py and original tests remain frozen at
Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. The intake verifies
eight exact source/dependency Git blobs and actual locked Kickmix 0.1 Simulator.

| Source definition | Native owner |
|---|---|
| TwoBitFFFT fields and signature | Ecosystem _fermionic_component_library.py converts stored metadata to the existing Algorithms Program with caller-owned x/y bit ports. |
| build_composite_bloq | Existing algorithms/primitives/fermionic_fourier.py owns the literal fifteen-call circuit and exact native inverse. The constructor reuses its helper graph and replaces its Rz metadata through the accepted Compiler rotation factory. |
| eps | Original precision stays on the serialized declaration and independent source rotation model. It does not change numerical action. |
| Symbolic k/n | Existing symbolic rotation and normal Circuit parameter binding handle both numerator and denominator expressions. |
| build_call_graph | Existing source_two_bit_fermionic_fourier_calls and cost retain precision and separate analytical rotation classification. |
| _fkn_matrix and my_tensors | Existing Simulation matrix owner retains the analytic tensor and adjoint convention. |
| Example, drawing and DocSpec | Existing catalog example/drawing owners are reused. The unchanged stored example resolves with default metadata. |

Interop owns conversion because the executable owner is an algorithm. Compiler
retains its transformation-only boundary. No source objects cross into native
IR and no primitive body or numerical backend is duplicated.

## Literal execution and source defects

The source decomposition ignores is_adjoint. Stored source constructors retain
that behavior for both flag values. The independent native adjoint remains an
exact inverse. The analytic tensor honors is_adjoint and differs from the
forward decomposition by exp(-i*pi*k/n). Existing exact complex tests preserve
that scalar explicitly instead of comparing only up to global phase.

Stored depths zero, one and two retain the same Program. Per-call producer,
port and operand traces are compared with the immutable source. Full bound
streams match the accepted algorithm, including reversed physical binding.
Complete complex operators match source decomposition across signed k/n cases.
Existing spectator, inverse, drawing, resource and append evidence is reused.
There are no private work wires, measurements or feedback. The only applicable
measurement history is empty and the clean-work condition is vacuous.

Actual execution always has one Rz, three S, six H, three CX, one T and one
T-dagger on two wires. The independent source model combines T/T-dagger into
two T and classifies its Rz as Clifford, T or arbitrary rotation according to
angle and precision. Literal instructions and model estimates stay separate.
The original n=0 rejection is retained. Negative n is accepted by the frozen
source and literal native constructor.

## Acceptance receipts

The concentrated affected regression passes 198 cases. Local integration
passes 145 cases. The unchanged original suite passes seventeen cases with
one original expected failure. Strict checking covers both changed Python
files. Black, Ruff, Ecosystem scope, language, architecture, multilevel IR and
public API checks pass. The dependency receipt retains exactly the existing
720 findings. Receipts use prefix g2-fermionic-constructor and date 20261123.
