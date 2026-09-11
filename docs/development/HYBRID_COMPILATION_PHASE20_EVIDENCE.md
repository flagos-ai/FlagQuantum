# Phase 20 detection-event temporal-decoder evidence

Date: 2026-09-09

## Accepted profile

`RepetitionTemporalDecoder` consumes every syndrome round available at a local
Runtime decision and validates that each recorded detection event matches the
change from the preceding syndrome. It waits for two consecutive observations
of the same non-zero syndrome. The second observation has no new detection
event and confirms a persistent data syndrome. An isolated readout excursion
returns to the prior syndrome with a paired temporal detection event and does
not trigger correction.

The decoder can drive either a physical X or a Pauli-frame X. This adds one
round of decision latency: single-data errors through the penultimate round can
be confirmed in the fixed-round experiment, while a terminal-round onset is
recorded but deliberately left unconfirmed.

## Evidence

| Gate | Result |
| --- | --- |
| Persistent non-zero syndrome is ignored once and corrected on confirmation | pass |
| Isolated readout excursion plus return event produces no correction | pass |
| Inconsistent syndrome/detection-event history fails closed | pass |
| Every data wire is corrected when injected in each confirmable round | pass |
| Physical-X and Pauli-frame-X temporal policies agree on decoded data | pass |
| Terminal-round onset remains visible with no executed correction | pass |
| Seeded 10% syndrome-readout profile emits fewer temporal than immediate actions | pass |

## Claim boundary

The two-round rule is a bounded causal reference, not a maximum-likelihood or
minimum-weight perfect-matching decoder. Repeated readout faults can mimic a
persistent data syndrome, and terminal-round errors lack a confirmation round.
The seeded finite-shot comparison characterizes one checked profile only; it
does not establish logical-error suppression, confidence intervals, a
threshold, general measurement-error tolerance, surface-code support,
controller latency, provider hardware, scalability, performance, or fault
tolerance.
