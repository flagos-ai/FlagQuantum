# FlagQuantum IR Phase 3 Batch F implementation review

Updated: 2026-09-03

## Result

Batch F implements a private, explicit opt-in shadow-comparison harness. Every accepted
invocation executes the legacy path first and returns that exact legacy observation as
the only authoritative result. The candidate result is never exposed by the outcome.

The closed mismatch taxonomy distinguishes status, result type, shape, value, semantic
identity, candidate failure and limit breach. Comparison evidence contains only bounded
timing and size fields plus SHA-256 identities; raw inputs, results and exception text
are excluded.

The harness requires an explicit immutable policy and kill switch. Comparison count,
input size, evidence size, candidate time and mismatch count are bounded. The switch is
thread-safe, one-way and first-reason-wins. Limit breaches preserve legacy authority and
stop later candidate execution.

There is no environment-variable activation, import hook, global singleton, background
thread, telemetry export, provider/network access, public export, `fq.run` integration,
default-path change or legacy retirement.

## Evidence

- Authorization, taxonomy, ordering, legacy authority, privacy, limit, kill-switch,
  concurrency, immutability and private-namespace tests pass.
- Anonymous fixtures pin evidence identities across Python hash seeds.
- The 10/100/1K/10K match-only baseline is recorded in
  `contracts/ir-phase3-batch-f-performance-baseline.json`.
- At 10K explicit comparisons, observed p95 was 158.639725 ms and peak traced host
  memory was 5,751 bytes.

The measurements are environment-specific observations, not a public SLA. Proposed
private regression envelopes remain unapproved in
`tests/fixtures/internal_ir/phase3_batch_f_performance_budget_candidate.json`.

## Remaining closed surfaces

Batch F does not authorize production or default-path shadowing, background sampling,
telemetry, provider SDKs, network calls, deployment migration, public API changes,
legacy retirement, Batch G deployment compatibility, Batch H, or Phase 3 exit.

## Review decision requested

Approve only the private Batch F performance budget and its future machine regression
gate with:

```text
approve IR-PHASE3-BATCH-F-PERFORMANCE-BUDGET
```

Batch F exit and Batch G remain separately gated.
