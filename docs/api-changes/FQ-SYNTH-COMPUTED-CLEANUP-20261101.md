# Computed register cleanup and storage reuse

Date: 2026-11-01. Continue sixth-plan resource-lifecycle dependencies after
#990, merged at 74231c735 with 57 successful checks and three configured skips.
This contract changes only private Compiler construction records. User Python,
native Qmod, public signatures, types, report fields and execution are unchanged.

## Closed operation and proof

Admit qreg.uncompute(effect, destination) -> next_effect with no attributes or
regions. A destination must be a non-Const register initialized by a direct
qreg.compute in the same verified body. Since that compute, neither its result
nor its direct source registers may have been written. Borrowing them through
verified compute/phase/Const calls preserves the proof; a gate write, even a
basis-diagonal or cancelling pair, invalidates it conservatively. An uncompute
of a dependency also invalidates dependent results' cleanup proofs. Initial
allocate/initialize and arbitrary prepared states do not create this proof.

Uncompute transitions initialized computed storage back to fresh proven-zero
storage. Its reference and physical allocation remain owned and available for
reinitialization or a new computation. Reads, double uncompute and returns
while fresh fail under the existing initialization contract. All declared
outputs still return initialized in declaration order. This is neither general
Input consumption nor dropping, freeing, measuring or erasing an arbitrary
state. It supplies a prerequisite for later local/within scope contracts.

## Exact selected inverse and accounting

Retain the actual selected value-computation ArithmeticCandidate and reverse
that circuit's admitted exact X/CX/CCX gates, including its rematerialization
and cleanup. Do not synthesize another implementation as an inverse. Reject
non-admitted gate payloads. Reuse the same physical inputs, result and zero work
tail; intervening segments clean their work and preserve required basis values.
Reverse existing allocation events with compute/uncompute actions exchanged;
derive inverse rematerialization counts from those events. Keep both actual
forward and inverse segments in the existing candidate, include inverse gates
in whole-program operation budgets and final resource estimates, and retain
the same peak workspace. No new evidence or scheduling record type is needed.

Compiler independently verifies every helper, then expands existing qreg.call
and verifies the complete flat root. Direct compute/uncompute within a helper
can compose; a caller cannot infer cleanup authority from an arbitrary helper
Output. References, effect order, widths, node/visit and call budgets retain
their current checks. Unsupported native syntax still fails before planning.

## Acceptance

Check all sixteen two-bit inputs, uniform complex states and external reference
entanglement; compute a flag, phase it, uncompute it and reuse its same physical
output storage. Verify clean internal work, exact absolute/relative phases and
interference under retain/recompute and width/depth selection. Include repeated
aliases, widened outputs, classical/zero arithmetic and nested dependencies.
Prove actual inverse instructions/events and budgets, mutation/unknown-proof/
Const/uninitialized refusal, closed malformed operations and stale effects.
Swap explicit cleanup with an existing directly borrowed phase producer behind
an unchanged Program/run/named-reader consumer. Record real costs and a private
contributor example, alongside a complete unchanged public Python user example.

No native locals/free/drop/within syntax, physical-pool release, generic unitary
inverse/control, new public role class, target relaxation, all-corpus or release
claim is admitted. Original sources/licenses/hashes and the four-file admitted
cohort remain unchanged; independent trials stay 0/2, release_authorized=false.
