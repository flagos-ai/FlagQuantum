# Explicit matrix and opaque Clifford constructor ownership

Reference: unchanged Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`, complete
`for_testing/matrix_gate.py` and `bookkeeping/arbitrary_clifford.py`.

## Interface fixed before implementation

Compiler formats consumes the original concrete `bitsize`, numeric `matrix`
and optional `atol` fields. Matrix storage becomes immutable built-in complex
rows. The existing Core matrix validator owns physical admission; Compiler
neither samples a matrix nor runs a numerical kernel. The single `q` register
uses the source Signature.build convention: QBit at width one, QAny otherwise.
Packed bits are split in MSB order around one existing explicit `any` leaf,
then joined without private work. Matrix entries retain their complete phase.
No gate synthesis or Clifford classification is inferred from a unitary.

The source matrix constructor's NumPy relative/absolute tolerance can admit
inputs outside Core's fixed absolute 1e-10 unitary bound. Those retain their
native refusal; `atol` cannot weaken the existing execution contract. The
zero-width source scalar remains outside physical Program matrix admission
and the pending registerless scalar proposal is not implemented here.

The opaque ArbitraryClifford consumes exactly `n` and delegates its signature
to the existing resource owner. Its FunctionRecord has no blocks. It is a
resource declaration, never an identity or guessed Clifford circuit. Concrete
and symbolic width declarations retain the existing Core validation and
physical refusal. The independent source model remains one Clifford and width
`n`; no primitive stream, amplitude or workspace result exists for this leaf.

Both source files already have accepted numerical/resource receipts. This
batch connects their surviving serialized constructor metadata, preserving
source-only and protected boundaries without counting their parents again.
Users bind an executable matrix Program and retain `circuit.append(op)`.

## Validation and remaining boundaries

The matrix constructor is exercised on widths 1/2/3, three original random
seeds, both actual source directions and serialized depths 0/1/2. Every matrix
entry retains exact bits; all complex basis columns include reversed logical
wires and a spectator. Native resolution blocks the source package and normal
append/JSON execution blocks both source SDKs. Matrix payloads are copied,
not aliased. Invalid shapes, nonnumeric values, nonfinite tolerance, wrong
ports and out-of-contract unitaries fail explicitly. There is no workspace or
measurement branch in this explicit leaf view.

Opaque widths 0/1/3 and both assumed/unconstrained symbols retain bodyless
declarations at all three source serialization depths. Execution refuses
missing bodies. The unchanged symbolic wire codec drops symbol assumptions;
the native declaration preserves the actual decoded symbol, rather than
inventing the lost assumptions. Resource checks use the original declared
width and independent one-Clifford model, separate from execution metadata.

The source model/tensor/random/adjoint receipts for the entire two files are
reused after frozen verification. One matrix is one actual primitive, zero
private work, and source width `bitsize`; the source has no QEC model. One
ArbitraryClifford is one modeled Clifford and width `n`, with no actual
primitive inventory or numerical amplitude claim. Generic external controls,
explicit metaclass inverses and zero-port physical execution remain scoped to
their existing owners or pending contracts.

The complete constructor inventory now has 200 Compiler keys, 39 Ecosystem
keys and 201 distinct keys out of 303. The 102 unmatched keys comprise 94
concrete classes, five abstract interfaces, two graph/descriptor special cases
and one original module-valued key. These totals are registration facts;
whole-library parent acceptance is unchanged.

The final batch passes 195 concentrated cases without skips in each actual
Cirq 1.7.0/Qiskit 2.5.0 and Cirq 1.6.1/Qiskit 2.0.0 profile, plus 145 local
integration cases. Six files pass strict typing, Black and Ruff. Team scope,
English/local links, architecture, internal IR and API checks pass. Qualtran
0.7.0 and the real Kickmix 0.1 Simulator are asserted. Seven complete original
implementation/test/dependency files match the locked CSV Git blob identities.
All 120 preexisting bookkeeping Programs and bindings remain identical to
`06151dd2f`. Existing full source numerical/resource receipts are reused;
this is not a GPU or complete repository test claim.
