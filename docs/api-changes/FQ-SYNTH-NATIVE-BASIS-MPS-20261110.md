# Native exact computational-basis MPS planning

Standing local implementation authorization covers the retained 50-control MCX
program. Its compiler capture and clean-workspace lowering are now shared with
the native 7- and 15-control programs. Public synthesis must still produce a
checked plan without pretending that a 64-wire statevector fits local memory.

Reuse the existing Runtime exact-MPS executor and native MPS result/marginal
reader. Add a Runtime-owned structural proof for a closed raw CircuitIR starting
in the implicit zero state: every instruction is a correctly sized I, X, Z or
CCX, with no parameters, matrix, parameter slots or instruction metadata.
These operations preserve computational-basis product states, including absolute
sign, and reuse the existing exact product-control executor path with stored
bond ranks one. CX/SWAP remain excluded: their exact SVD executor can retain
zero singular directions, so mathematical rank one alone does not certify
the actual tensor footprint. Keep the proved instruction vocabulary during
Runtime planning; floating Euler rewrites could introduce small components. No numerical execution is used to
make the planning decision. Custom Circuit initial states do not qualify for this
proof. Channels, dynamic instructions, unproven matrices, rotations, Hadamards,
distributed execution and approximate execution are not admitted by it.

Native synthesis may select this exact plan for more than 32 physical wires
only after its chosen submitted circuit passes the proof. Record the profile
and rank bound in the synthesis report. This is selection before execution,
with no fallback or approximation. Runtime independently recomputes the proof,
and Services independently checks the sealed execution circuit and state-buffer
estimate. Preserve the existing independent-interval proof for other MPS plans.
Resource, state-buffer and named-read budgets remain enforced. The buffer
estimate is not a process peak-memory claim. No dense full-state output is
materialized by the example or acceptance tests.

Acceptance includes the unmodified 50-control source through its existing
test-only compiler/basis-MPS reference and the public native Program path,
original width and all 13 scratch bits, active/near-active/nonzero-target cases,
absolute product-state sign and exact tensors versus an independent Circuit.
Forbid dense conversion, verify bounded named target readout and read refusal
for the full control register, and test proof rejection, state memory ceilings,
workspace/operation/depth/qubit budgets and saved-plan validation. No new
numerical core or public signature is introduced. Local implementation is
authorized; release and scalability claims remain unauthorized.
