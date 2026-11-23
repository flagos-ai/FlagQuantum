# Atomic scalar constructor closure

## Source scope and ownership

The complete locked basic_gates/global_phase.py constructor file is adapted
from Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389.
Compiler formats/legacy_scalar_factories.py owns only constructor validation
and composition. It reuses the existing scalar_phase_ir primitive recipe,
typed component builder, existing symbolic parameter translator and the
existing exact source binding evidence. The original source is unchanged.

GlobalPhase retains zero logical ports. A nonzero phase allocates one clean
private carrier, applies X/phase/X/phase, and frees the restored carrier.
Exponent zero emits no gates and allocates no carrier. Symbolic exponents bind
through existing approved parameter expressions, including nonlinear ones.
Precision is retained in the original serialized source and independent model;
it does not change numerical action. The original positive control is a
ZPowGate and uses the already accepted rotation constructor. Generic serialized
outer control and adjoint wrapper dispatch remain separate integration work.

The user path remains circuit.append(op), using the bound Program primitive
stream. Source tensor phases are compared without quotienting global phase.
The existing source-binding audit covers coefficient construction, source SDK
interop, nested controls and separate numerical/analytical representations.

## Verification and resource separation

Named original objects pass through the unchanged source serializer and native
resolver. Both directions, positive-control dependency, complex64/128 coherent
spectators, zero private carrier, typed ownership and nonlinear symbolic binding
are checked. The actual carrier gate order and operands are asserted explicitly.
Existing accepted global phase source-binding, primitive and symbolic evidence
is rerun alongside rotation and controlled-constructor dependencies.

| Case | Actual primitive/work inventory | Original analytical model |
|---|---|---|
| Nonzero scalar | Two X, two phase instructions, one clean carrier restored and freed | Zero QEC gates and zero logical qubits |
| Zero scalar | Zero gates and zero carrier allocations | Zero QEC gates and zero logical qubits |
| Positive controlled scalar | One phase instruction on the control, zero private work | Original ZPow rotation model, kept separate from synthesis estimates |

The original scalar is atomic and supplies no physical primitive decomposition.
The four-gate carrier recipe is the existing accepted native materialization,
not an upstream primitive gate count. No measurement occurs in this constructor,
so there are no measurement branches to enumerate. Parameterized circuits use
the existing symbolic execution path and fail normally on unresolved parameters.

The related regression passes 362 cases, with six optional Qiskit interop
cases skipped because that SDK is absent from this execution environment.
Existing accepted evidence remains distinct from these skips. Local integration
passes 145 cases. The unchanged complete original test file passes eight cases
with three original atomic/decomposition skips. Four frozen blobs and the
native Kickmix 0.1 Simulator extension are verified. Both changed Python files
pass strict typing, Black and Ruff. Compiler scope, language, architecture, IR
and API checks pass. Dependency findings retain the identical 720-line baseline.

## Integration boundaries

One complete constructor identifier closes, bringing complete built-in coverage
to 65/303 with 238 remaining. ControlledViaAnd generic child dispatch,
Controlled/Adjoint/Power, arbitrary-target Pauli, partial OnEach and permissive
non-signed sign conversion remain open. Existing source codec defects, pending
Proposals 116-119 and whole-repository/GPU/performance gaps remain separately
recorded. Evidence uses ftqc-reference-env/g2-scalar-constructor-*.
