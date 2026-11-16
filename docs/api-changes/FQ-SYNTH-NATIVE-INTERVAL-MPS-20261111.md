# Native exact interval MPS admission

Status: admission implemented and locally verified. Native corpus acceptance
remains 206/216; tensor hypercontraction itself is not yet accepted. Four new
admission tests and twenty existing basis-MPS/zero-walk regressions pass;
strict typing, Black and Ruff pass for the two changed Python files.
Classification: `single_device_fast_path`.

The native qfunc producer currently requests statevector execution except for
proved computational-basis circuits wider than 32 bits. The original tensor
hypercontraction program instead uses exact MPS after a checked zero-entry walk
proof. Native high-level programs with many idle fields and short disjoint
active intervals should reuse the same Runtime-owned exact interval bond proof,
rather than requiring an impossible full-state allocation.

Reuse `runtime.planner.estimates.exact_interval_mps_bond` on selected submitted
IR gate supports for native Python programs wider than 32 bits. Only choose MPS
when this existing helper returns a finite bound; it rejects intervals wider
than twelve physical sites. Keep the existing computational-basis proof first.
The shared `_complete` authority still plans, validates memory and capability,
uses complex128 CPU execution, forbids approximation and backend fallback, and
validates resources. Record the exact interval profile and intermediate bond
bound in the report. This changes no public signature, export, runtime proof,
numerical kernel, width/effect cap or caller budget.

Acceptance must include an 83-bit native program with a nontrivial four-bit
clock and two independent spin bits, exact MPS versus an independent small
Circuit over its entire occupied support, support probability one, no truncated
bonds, original memory-budget refusal and an overlapping wide connected gate
interval that remains refused. Small native statevector execution and existing
basis-MPS behavior must remain unchanged. Confirm no Runtime or Simulation
changes are needed.

This admission alone does not accept the tensor hypercontraction example. Its
source's initialized-subspace cancellation requires a separate independent
semantic proof. Nonzero auxiliary-record input cannot be advertised as supported
by the zero-entry reduction. Preserve source and prior baselines, then run the
fresh 216-original and approved-six replay with any changed production pin
explicitly identified before crediting the example.
