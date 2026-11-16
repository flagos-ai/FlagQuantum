# Borrowed Fourier transforms under scalar coherent control

Status: implementation contract; accepted native corpus remains 207/216.
Classification: `single_device_fast_path`.

Thermal's inverse block contains a complete-register inverse Fourier transform.
The existing native borrowed callback captures this as `qreg.inverse_qft`, but
the shared controlled-effect adapter does not currently translate it. Replacing
the transform with a dense matrix is semantically possible but has unacceptable
construction and workspace costs for the retained fifty-cycle channel.

For an initialized scalar control, translate existing `qreg.qft` and
`qreg.inverse_qft` effects into the authoritative `qreg.control_qft` effect,
retaining the inverse flag. Reuse existing register verification, inverse rules,
Fourier lowering, resource accounting and execution. Introduce no public API,
effect opcode, schema or numerical kernel. General indexed controls in wider
registers remain outside this narrow translation and retain explicit refusal.

Acceptance: forward and inverse borrowed callbacks, arbitrary complex target
preparation, coherent scalar controls, absolute phase and both control branches
must match an independent analytic Fourier matrix Circuit. Reject aliases and
operation/work/memory budget violations. Existing source replay outcomes and
frozen resources must remain unchanged at the next complete thermal batch replay.
This enabling change alone earns no additional native corpus acceptance.
