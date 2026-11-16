# Framework exception export source audit

This batch follows Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The exception initializer's frozen
Git blob is `7603935118da18e88de4b01ef497291de111e762`.

| Source export | Native behavior and owner |
| --- | --- |
| `DecomposeTypeError` | The typed component traversal retains a materialized primitive leaf when expansion is inapplicable. The stable API uses Python `TypeError` for invalid argument types. |
| `DecomposeNotImplementedError` | A missing requested stable operation is represented by `CapabilityError`, which is catchable as both `NotImplementedError` and `FlagQuantumError`. |
| `BloqError` | Compiler component binding reports invalid ports with `RegisterBindingError`, a `ValueError`. The underlying native verifier retains structured graph diagnostics. |
| `DidNotFlattenAnythingError` | A selective one-pass expansion with no eligible call raises `NoComponentExpansionError`, a `ValueError`. Repeated expansion catches that signal and returns the original verified program when no pass made progress. |

The source test verifies the export list and identity of all four definitions.
It exercises the real source exceptions for an atomic gate, a missing rule,
a missing port binding, and a no-change flatten pass. The native counterpart
checks atomic-leaf retention, missing-port diagnostics, no-change termination
and stable capability catchability when attempting an adjoint of a measured
circuit. The measurement and original instruction list remain available after
the failed adjoint request.

Source Bloqs can execute a lazy decomposition callback while expanding a
graph. Native traversal operates over verified materialized definitions.
The tensor flattening regressions preserve the source exception distinction
and independently verify propagation at the native callback boundary.
The stable exception namespace is governed by
`contracts/errors-module-boundary-v1-candidate.json`.

These checks cover control flow and ownership. They produce no physical
instructions or model estimates. Existing composition and graph regressions
verify successful circuit construction separately.

The broader composition contract reproduces the recorded baseline failure
in the QFT instruction-order fixture
`test_compose_expands_to_the_hand_built_program`. The exception regressions
run independently of that fixture.
