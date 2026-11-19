# Complete local resource-estimator interoperability notebook

Q-0915/Q-0916 cover the full frozen notebook and unchanged original test at
Qualtran v0.7.0 revision `bcb17bb25360ed68ce3bc084645db3e929799389`.
The notebook blob is `25b2ba4cb6b9073444016e3c89e72f9de4a0c2e8`.
The native notebook is `examples/notebooks/physical_resource_interop.ipynb`.

All eight original cells retain their order, type and source-cell index. One
additional cell records license, dependencies and scientific scope. The native
ModularExponentiation metadata and existing signature-graph renderer replace
the source component and diagram. The example declares the same 32-bit exponent
and result, base 8 and modulus 221. Its source call chain declares e controlled
multipliers, each with two scaled adds and n controlled swaps. Each scaled add
has n*(5*n+2) forward ANDs. Existing legacy gate-budget folding converts these
categories to 1,334,272 T gates. The helper's count export is independently
compared with unchanged source t_complexity at three distinct word widths.

The source notebook exports 64 signature qubits. Its full resource width is
195 qubits. The native example explicitly reproduces the signature-only estimate
and labels this undercount. The reusable converter instead accepts any existing
ModeledResources, so callers can supply full peak width. It also accepts explicit
rotation depth and preserves the source default of rotation count. It exports
exactly the original four known-count fields. SDK defaults retain zero CCZ,
CCiX and measurement counts. No measurement-depth fidelity is inferred.

`ecosystem/resource_estimator.py` owns conversion to the external schema. It
reuses the compiler's legacy T-category convention and rejects nonconcrete or
negative output counts. It imports no optional estimator, invokes no remote
service and changes no shared adapter registry or protected API. The optional
SDK and widget objects remain in the example, outside Core, Compiler, Runtime
and Simulation. Native circuit construction continues through circuit.append.

The isolated reference environment uses qsharp 1.9.0, qsharp-widgets 1.9.0,
anywidget 0.9.18 and psygnal 0.14.0. These packages are local optional tutorial
dependencies, with their own installed license metadata retained in the reference
directory. The native package gains no runtime dependency. Replacing the SDK
requires only consuming the same known-count dictionary. Removing the optional
tutorial requires no core migration. This boundary exists specifically to
reproduce the complete source's physical estimates and SpaceChart payload.

All four original hardware configurations execute locally, including the two
Floquet-code Majorana presets. Their complete result dictionaries match exactly,
including physical counts, runtime, distillation, formatting and assumptions.
SpaceChart receives an identical estimate payload. Native execution blocks
Qualtran, Cirq and OpenFermion imports. The unchanged original notebook test runs
in an isolated Jupyter kernel with the same fixed optional SDK.

Physical estimator results are analytical external-model outputs. The example
executes no full 32-bit factoring statevector and makes no new amplitude,
measurement-history or workspace-cleanup claim. Previously accepted modular
arithmetic execution evidence remains separate. Rendering of the Microsoft
widget is owned by that explicitly selected external presentation dependency.

All 451 concentrated resource regressions pass without skips. The converter,
its tests, notebook test, extracted notebook script and six updated reference
test modules pass strict mypy, Black and Ruff. Native component signature
rendering is inspected as a PNG. Ecosystem team scope, architecture, IR and
protected API checks pass. Optional SDK imports in the new notebook test and
reference modules are guarded for dependency-free collection. The isolated
qualification environment confirms all dependencies exist, so these guards
skip none of the acceptance cases.

The full-repository dependency-policy scan remains failing on older optional
imports outside this batch. Its complete diagnostics and batch-path census are
recorded in `g2-resource-estimator-interop-dependencies.log` and
`g2-resource-estimator-interop-dependency-scope.json` in the reference environment.
Every path changed by this batch has zero dependency-policy findings. This is a
scoped pass, not a whole-repository dependency-policy pass. The remaining
collection debt must be resolved before final G2 acceptance.
