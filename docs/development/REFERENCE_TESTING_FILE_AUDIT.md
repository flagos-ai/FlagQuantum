# Register example testing source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen files are
`qualtran/conftest.py` (`9699d1a2feb3497898606342ef711f94f2a8f596`),
`qualtran/testing.py` (`e06e8f1f785390053dfea68da8178132d38185a1`),
and `qualtran/testing_test.py` (`5f34d93ee9c226abf03331b2c47dee6ab1221ad6`).

The test-only register example fixture uses an owned program factory, an
explicit lowering function, and a separate instruction-count declaration.
The five checks follow the order of the source pytest wrappers.

| Check | FlagQuantum owner and behavior |
| --- | --- |
| Make | The fixture checks that the factory returns a register program. |
| Decompose | The program verifier checks the signature, value producers, operands, uses, and returns before validating the lowered circuit. An atomic fixture may omit physical lowering. |
| Counts | The declared counts must agree with the lowered instruction stream. Missing independent evidence is reported explicitly. |
| Serialize | The existing circuit codec restores wire and instruction data through a JSON roundtrip. |
| Quantum typing | The program verifier and register compatibility checks cover structural validity and loose, any, and strict compatibility. A loose mismatch fails and stricter-only disagreements remain unverified. |

The reference wrapper for quantum typing can consume a failure result and
return normally. A probe against the frozen source reproduces this behavior.
The owned pytest adapter surfaces the failure. The source count and serialization
expected failures stay in the reference tests, including the legacy serializer,
EC point, and XOR instance exception groups. Owned tests use explicit imports.
The existing notebook runner remains responsible for notebook execution.

## Verification

The unchanged source suite passes all 14 cases. The native fixtures cover
the original parallel decomposition, parent and boundary signatures, graph
membership and value reuse, factory outcomes, all four dtype combinations,
and valid and invalid classical action. A three-wire append fixture compares
all eight complex basis columns and independently asserts the three X
instructions. The synthetic source atom's opaque cost declaration is separate.

The signed three-bit adder checks all 64 input pairs against the original
classical callback and independently against its physical decomposition.
The four complete measurement histories preserve the expected complex
amplitudes and leave the two reserved work wires in zero. The physical stream
contains eight T or T-adjoint instructions and two measurements. The source
model declares two And operations and two measurements.

## Open scope

The generic source testing helper remains open for arbitrary source
example-object serialization, low-level wire-symbol comparison, and arbitrary
source typed-graph fixtures. The native checks above cover owned register
programs and the explicit circuit serialization contract.
