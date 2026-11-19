# Named runtime execution inputs approval

On 2026-11-12 the human user explicitly replied "Approved" to the concrete
[proposal](../development/EXECUTION_RUNTIME_INPUTS_PROPOSAL.md) and executable
contract fake. The approval adds only a keyword-only
`inputs: Mapping[str, float | Tensor] | None = None` after `initial_state` on
`fq.run`, plus the bounded `runtime_inputs` plan extension. Naming review
separates named program arguments from the entry quantum state and execution
options. Existing Parameter and ParameterExpression remain the declarations.

The initial slice is exact unbatched CPU PyTorch statevector execution with
finite real scalar angle inputs, gradients and compatible plan replay. External
initial states cannot be combined with named inputs in this slice. Other
engines, devices, dynamic programs, synthesis products and remote execution are
not admitted. fq.plan and other public signatures remain unchanged.

The change is additive: existing calls and snapshots remain intact; callers may
continue using Circuit.bind_parameters. There is no deprecation or removal.
The integration contract and fake are recorded before Runtime implementation.
Implementation must pass replacement, public conformance and persistence tests.
This approval does not certify maturity, performance or release readiness.
