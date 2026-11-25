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

## First complete outcome: not adopted

The declaration was published at
`9e23002afc78de142e113f6e7e81e365e6a843ff` before the first full control.
[The exact archive](evidence/cirq-cz-invariant-pruning.json) retains the proposed
source, balanced driver, all observations and reconstruction hashes. All 120
pairs and 240 independent acceptances preserve normalized outputs, evaluator
metrics and adapter audits. The supervisor completed once with status zero.
No local tests or artifact transfers overlapped the measurements.

| Inspected case | Baseline / proposed median | Observed reduction | Compiler-stage Python traced peak, baseline / proposed |
| --- | ---: | ---: | ---: |
| 6-wire qft, grid | 91.622 / 90.736 ms | 0.97% | 1,917,926 / 2,073,198 bytes |
| 6-wire repetition, ring | 116.636 / 113.780 ms | 2.45% | 2,541,939 / 2,456,069 bytes |
| 12-wire qft, grid | 565.163 / 554.395 ms | 1.91% | 10,116,699 / 10,116,693 bytes |

No case meets the declared 5% median reduction; six-wire QFT's observed peak
increases. These shared-host observations do not establish statistical
significance, a universal performance effect or process-memory qualification.
They do not satisfy the fixed admission, so the candidate is not adopted.
Do not rerun or revise thresholds/cases to rescue this implementation.

## Reducing-block and classification controls

The six historical stage cases are retained, with an eight-local-rotation prefix
added to each of the four reducing cases to exercise the 43-operation admission
bound. Their exact source is fixed before those stage observations and retained.
All ten cases preserve exact proposed outputs and rejection reasons. Returned
operators are checked independently with NumPy at absolute tolerance `1e-11` up
to one global phase. This development-only reference is not a production dependency.
Two warmups and twenty retained alternating samples give:

| Stage case | Weyl calls, baseline / proposed | Stage median, baseline / proposed |
| --- | ---: | ---: |
| refocused_pi_x | 1 / 1 | 171.6 / 172.1 us |
| refocused_pi_x_with_frame | 1 / 1 | 194.9 / 195.1 us |
| quarter_turn_pauli_product | 1 / 1 | 327.3 / 327.1 us |
| mixed_wire_two_cz_reduction | 1 / 1 | 435.2 / 435.3 us |
| generic_two_cz_control | 1 / 1 | 203.4 / 203.8 us |
| generic_three_cz_control | 1 / 0 | 299.4 / 72.7 us |
| refocused_pi_x_admitted_locals | 1 / 1 | 268.0 / 287.5 us |
| refocused_pi_x_with_frame_admitted_locals | 1 / 1 | 279.1 / 297.4 us |
| quarter_turn_pauli_product_admitted_locals | 1 / 1 | 367.3 / 387.0 us |
| mixed_wire_two_cz_reduction_admitted_locals | 1 / 1 | 521.0 / 539.9 us |

The admitted reducing cases still need one Weyl decomposition and add invariant
work; the generic three-CZ control avoids Weyl entirely. This is a real local
tradeoff, not whole-SDK evidence of superiority. The first stage run also checks
320 fixed-seed, locally dressed zero/one/two/three-CZ operators with global phases
against a separate NumPy implementation of the inspected Cirq formulas. No
conservative refusal contradicts that reference, and no constructed reducing
operator is refused. This finite conformance is not an exhaustive proof of
floating-point boundary safety. Full tolerance-boundary, resource-precedence,
symbolic and metadata conformance remain unproven for this unadopted prototype.

The result supports keeping the existing production synthesis path. A faster
classification stage alone is insufficient to close the remaining delivered
compilation overhead. Native-CZ routing selection remains unqualified for the
default compiler. No formal campaign was started and no machine was restarted.
