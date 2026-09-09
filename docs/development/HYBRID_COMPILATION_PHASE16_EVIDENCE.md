# Phase 16 repetition-code memory evidence

Date: 2026-09-09

## Accepted profile

The experimental `flagquantum.qec` domain composes the private bounded hybrid
compiler and local dynamic Runtime into one three-data-qubit repetition-code
memory workflow. Two reusable ancillas measure adjacent Z-parity checks for a
fixed number of rounds. The compiled reference policy corrects zero or one
deterministically injected data-qubit X error.

The workflow returns typed, shot-resolved syndrome rounds, temporal detection
events, decoder decisions, final data bits, and logical-zero memory outcomes.
`Decoder` is replaceable for post-execution syndrome analysis; it is not yet a
real-time Runtime callback and does not replace the compiled feedback policy.

## Evidence

| Gate | Result |
| --- | --- |
| QEC concepts live under an independently owned `flagquantum.qec` domain | pass |
| The stable package root gains no exports | pass |
| Zero-error logical-zero memory remains `000` | pass |
| X on data wire 0 produces first syndrome `10` and is corrected | pass |
| X on data wire 1 produces first syndrome `11` and is corrected | pass |
| X on data wire 2 produces first syndrome `01` and is corrected | pass |
| Later syndrome rounds clear after reference feedback | pass |
| Syndrome transitions produce typed detection events | pass |
| Reference and batched trajectory strategies agree | pass |
| A structurally conforming analysis decoder replaces the reference analyzer | pass |
| Invalid wires, round counts, shots, and record invariants fail closed | pass |

## Claim boundary

This evidence establishes one noiseless, bounded, local repetition-code memory
experiment and a first QEC-owned data contract. It does not establish general
stabilizer-code support, realistic error models, logical-error suppression,
threshold behavior, real-time decoder integration, controller latency,
provider hardware, gradients, distributed execution, capacity, performance,
or fault tolerance.
