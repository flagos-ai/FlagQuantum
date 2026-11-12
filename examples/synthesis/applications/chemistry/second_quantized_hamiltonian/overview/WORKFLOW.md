# Second-quantized workflow acceptance

Review all seven source code cells: installation, imports, ordered fermionic
Hamiltonian, mapper and native ansatz, synthesis, variational minimization and
actual result readout. Use existing FlagQuantum algebra, observable and execution
APIs; no chemistry dependency or source importer is added.

Independently apply each ordered creation/annihilation monomial from right to
left, retaining parity signs. Verify every one of sixteen occupation columns
against public observable expectations. Compare complete complex states for the
zero profile, two nonzero profiles and actual best parameters against direct
Circuit X/RY/CX layers. No phase alignment is allowed. Check mode order and
zero workspace, plus width, operation, depth and memory budget rejection.

Preserve twelve zero initial angles. Execute 200 actual objective evaluations,
retain rejected candidate energies and reexecute the best parameters. Check the
reported energy against the independent occupation matrix. Record termination
as budget exhaustion. Keep known eigenvalues out of optimizer inputs.

Retain particle metadata (1, 1) as source context without claiming that the
ansatz enforces it. Compare actual total-particle populations against independent
Circuit probabilities and verify that more than one sector is populated.

Execute all four native Notebook cells in pytest and an actual Jupyter kernel.
Keep saved source outputs empty. Run native and comparison CLIs. Recheck all
216 originals plus six approved revisions, preserving existing semantic refusals.
