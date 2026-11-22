# Cirq characteristic-invariant CZ pruning control

This is a Compiler-owned `single_device_fast_path` proposal. The production
compiler remains unchanged until independent acceptance and resource review.

## Upstream mechanism and fixed proposal

Cirq's `num_cnots_required` and `_gamma` classify minimum CNOT/CZ counts from
characteristic-polynomial invariants, without KAK eigenvectors or synthesis.
The inspected Apache-2.0 source is
[`decompositions.py`](https://github.com/quantumlib/Cirq/blob/de651a7886d1a1383d595f95f36b93ed3ea91685/cirq-core/cirq/linalg/decompositions.py)
at revision `de651a7886d1a1383d595f95f36b93ed3ea91685`, SHA-256
`c3d1b2dfb998ddada8a29e4799999f9a8314bbedf2a1bd8d15f25329260df846`.
The source names Shende et al.'s characteristic-polynomial criteria. The pinned
upstream tests cover zero through three entanglers with random local dressing.
The prototype retains the Cirq copyright and Apache notice.

The proposed `_cz_blocks.py` SHA-256 is
`a926a6436f91fd07144d4e6541c54512cd4282eff440467d5418aca07e3b5336`.
It adapts those invariants to existing native list algebra and determinant code,
without NumPy or an external SDK dependency. It is a conservative refusal only:
a two-CZ block must be separated from both zero/one-CZ classes, or a three-CZ
block must be separated from the two-CZ class. Ambiguous cases retain full KAK.
The exclusion margin is `1e-6`, looser than upstream's default `1e-8`; it does
not change any accepted operator tolerance. Orthogonality must hold within
`1e-12` before this precheck is considered. Nonunitary or uncertain inputs keep
the original synthesis/error path.

The existing raw three-entangler/four-local-layer form has at most
`3 + 2 * 5 * 4 = 43` instructions. Pruning is allowed only when that full bound
fits the effective existing resource limit, preserving instruction-growth
refusal precedence. This is not Weyl preclassification followed by another Weyl
calculation. No input-dependent cache, public API, routing strategy, evidence
schema, dependency, budget or synthesis-acceptance change is proposed. Every
actual rewrite still uses existing synthesis and independent operator checks.

## Fixed complete-workflow control

The baseline is integrated main
`7d2e13c4160c0c1546d61a41551189fa11615465`; SDKBench stays unchanged and clean at
`9aa9260920984e2ca05bcb293abb885fccd9c144`. Use inspected nonce `01` repeated
32 times, widths 6/8/12, QAOA/Ising/hopping/QFT/repetition and
line/ring/grid/sparse: 60 cases each for ordinary-restored and target-artifact
workflows, or 120 pairs and 240 independent acceptances. Require identical
normalized operations/layout hashes, evaluator metrics and adapter audits.
Six-wire cases have complete-operator acceptance up to phase; larger cases have
three probes. This is inspected regression evidence, not held-out qualification.

Both implementations are AST-cloned functions installed through the same cached
module-attribute patch list, with equal lifetimes and restoration assertions.
Keep Python 3.12.14, one numerical thread and the existing garbage collector.
Do not overlap tests or artifact transfers with measurements.

After acceptance, use target-artifact QFT6/grid, repetition6/ring and QFT12/grid,
two warmups and nine retained alternating samples per variant. Separately trace
allocations around `adapter.execute` only, matching the timed compilation stage;
fixture/configuration preparation and independent evaluation remain outside that
scope. Earlier whole-observation traced peaks included bundle parsing and are
not comparable to these compiler-stage peaks. These are still Python traced
allocations, not process RSS or qualified memory measurements.

Require at least 5% lower median adapter time in all three timing cases and no
larger traced peak in any case before proposing adoption. Retain the first
completed run, every sample and failure. Do not tune cases, change thresholds or
repeat observations to chase a favorable result. Success here is necessary,
not sufficient: focused invariant, boundary, reducing-block, resource-precedence,
symbolic, metadata and independent-evidence tests remain required. In particular,
measure and disclose reducing-block overhead before any integration decision.

A separate 600-second process-group supervisor owns TERM/KILL cleanup and leaves
SDKBench's stage timer intact. Formal dedicated-host qualification remains
user-deferred. No machine, service, kernel or driver restart/change is permitted.
No SDK ranking, release, accelerator or scalability claim follows.

## Existing supported workflow

```python
import flagquantum as fq
from flagquantum.compiler import optimize
source = fq.Circuit(3).h(0).h(1).h(2).to_ir()
assert optimize(source).instructions == source.instructions
```

Expected: independent gates remain unchanged. The proposal supports existing
native target-artifact compilation; it introduces no public entry point.
