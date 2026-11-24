# Stored cost-function phasing constructor audit

## Scope and ownership

The complete frozen `rotations/phasing_via_cost_function.py` constructor composes
cost evaluation, quantum variable rotation and exact evaluation erasure. The
locked reference is Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Its source stays unchanged.
Classification is `single_device_fast_path`.

`formats/legacy_cost_phasing_factories.py` owns constructor adaptation and
per-compilation dependency recipes. It reuses the QVR constructors, original
Square declarations and `registers/phase/cost_function_component.py`. The existing
Program builder owns typed calls. Numerical execution and backend choice remain
outside Compiler. Normal execution retains `circuit.append(op)`.

Inputs and extra phase registers belong to the caller. Evaluation privately
produces RIGHT cost words. Each phase cost port must be THRU with matching width
and array shape. Words are MSB first. Exact erasure consumes the evaluator image
and releases its clean cost outputs. Component owners retain all private work,
measurement feedback and release responsibilities.

## Direct dependency closure

Both original five-bit Square examples retain the exact three-call root and
source operands at serialized depths zero and one. Their direct/gradient QVR
children reuse the accepted phase-power/scaled-add owners. Original Square has
no gate decomposition. Its forward and reverse constructors remain bodyless
native declarations. This preserves the original atomic boundary.

A caller may independently bind an executable evaluation implementation and
provide its exact inverse module. Small Square cases use the existing measured
column-compression owner and its explicit reverse program. The constructor does
not derive an inverse by reversing measured instructions. Complete resolved
functions key the default Square and QVR recipes. Unknown evaluators require an
exact-adjoint provider. Custom phase owners provide explicit cost-register names.
Child helper definitions remain owned by the surrounding resolver and are not
copied or renamed into a second dependency closure.

## Verification and boundaries

The focused tests compare source signatures, all root operands and source graph
connections. Replacement tests bind the actual constructor entry and require
identical primitive streams, feedback, allocation events, releases and outputs
to the previously accepted computation/phase/erasure Program. Full small complex
tensors and all applicable measurement histories are certified, including zero
workspace amplitudes. Existing inverse, spectator and ordinary append evidence
is reused through this exact physical equality.

Original parent validation checks width, shape and side, without requiring
nominal dtype equality. The existing native typed binder retains its own dtype
compatibility rules. Original-compatible Square unsigned/fractional cost views
are covered. Arbitrary incompatible dtype pairs, unresolved physical domains
and unprovided generic adjoints remain explicit boundaries. The constructor
cannot certify the numerical truth of an arbitrary caller-supplied inverse.

Actual primitive inventories and source analytical resource models are separate.
The original Square model estimates Toffolis while the original Square root has
no primitive stream. Native measured Square decompositions are not claimed as
source gate decompositions. Larger examples reuse prior bounded coherent and
resource evidence, rather than claiming an exhaustive large statevector run.

## Acceptance receipt, 2026-11-24

The concentrated regression passes 103 cases. Related local integration passes
145 cases. Unchanged original file tests pass ten cases, skip two and retain
29 expected failures. The source fixture verifies all four frozen file blobs,
Qualtran 0.7.0 and actual Kickmix 0.1 Simulator availability. Strict mypy checks
both new Python files. Black, Ruff, Compiler scope, language, architecture,
multi-level IR and API checks pass. Dependency analysis exactly matches the
720 pre-existing findings.

The initial direct pytest launch encountered a missing optional plugin. With
plugin autoload disabled, four recursive graph fixtures needed the existing
reference launcher's notebook marker registration. The corrected full run passes
all 103 cases. No source test was edited or deselected.

| Original five-bit example | Actual native evaluation/phase/erasure primitives | Independent original model |
| --- | --- | --- |
| Direct QVR | 160 T/Tdag, 394 CX, 10 phases, 40 measurements, 35 used wires | 40 Toffolis, 10 rotations, 15 logical wires |
| Gradient QVR | 1636 T/Tdag, 4062 CX, 409 measurements, 78 used wires | 520 Toffolis, 47 logical wires |

The resource probe reserves a deliberately oversized pool of 100 clean wires.
Total reserved capacities are 105 and 137. The used-wire counts in the table
exclude idle reserved wires. Conditional CZ and resets are retained in the
primitive inventory. These two original factory inventories match prior accepted
native lowering exactly.
