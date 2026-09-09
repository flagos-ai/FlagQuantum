# Phase 17 timed-error and Pauli-frame evidence

Date: 2026-09-09

## Accepted profile

The experimental repetition-memory workflow accepts a canonical, bounded
schedule of deterministic X-error events. Each event identifies a data wire
and a round and is injected at the start of that round before parity checks.
The same typed syndrome history supports two explicitly different modes:

- `compiled_lookup` executes immediate lookup feedback inside the lowered
  dynamic program;
- `offline_pauli_frame` executes no physical feedback, passes the complete
  history to a decoder, and applies its parity-reduced frame to final readout.

Every shot records executed feedback separately from the decoder's proposed
corrections and frame. This prevents offline analysis from being represented as
a real-time Runtime callback.

## Evidence

| Gate | Result |
| --- | --- |
| Error schedules sort deterministically and reject duplicate round/wire events | pass |
| An error outside the configured round range fails before compilation | pass |
| Each data wire can receive a single X error in any of three rounds | pass |
| Compiled feedback restores every verified single-error data state | pass |
| Offline terminal-syndrome decoding restores every verified single-error readout | pass |
| Decoder consumes the complete dense syndrome history | pass |
| Executed feedback and decoder recommendations have distinct typed records | pass |
| Detection events expose syndrome onset and compiled-feedback clearance | pass |
| Two same-round data errors produce an explicit logical failure in both modes | pass |
| Reference and batched trajectory strategies agree for compiled feedback | pass |
| Invalid schedules, modes, histories, frames, and result records fail closed | pass |

## Claim boundary

This evidence establishes deterministic error scheduling and offline
Pauli-frame semantics for one small repetition code. It does not establish a
stochastic channel, measurement errors, circuit-level noise, repeated noisy
logical-error-rate estimates, logical suppression, threshold behavior,
real-time decoder invocation, controller latency, provider hardware, general
stabilizer codes, gradients, distributed execution, capacity, performance, or
fault tolerance.
