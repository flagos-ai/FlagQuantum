# IR Phase 2 Batch B completion review

Status: implementation and approved performance gate complete; owner exit approval pending.

## Delivered

- A private, opt-in target-decomposition pass lowers all 30 certified static unitary opcodes to `rx`, `ry`, `rz`, and `cx`.
- State and trainable-gradient differential checks cover rewritten operations.
- Unsupported channels and custom operations fail closed.
- Program identity, pipeline digest, target legality, idempotence, and deterministic output are tested.
- The approved internal CPU budget is now an executable fail-closed gate at 10, 100, 1,000, and 10,000 source gates.
- Stable public APIs, the default execution path, emitters, provider code generation, and legacy retirement remain unchanged.

## Performance evidence

The approved 5-iteration, 2-warmup Docker validation passed every latency, memory, and determinism threshold. At 10,000 source gates, the pipeline emitted 60,000 native gates with 2,190.197026 ms p95 latency against a 2,700 ms budget, and 123,472,479 bytes peak traced host memory against a 167,772,160-byte budget.

These values are internal regression budgets, not public service-level commitments.

## Batch C proposed boundary

Batch C should add private placement and routing only after separate owner approval. Its initial scope is:

- explicit logical-to-physical qubit mapping;
- coupling-map and directed-edge legality;
- deterministic placement and routing metadata;
- asymmetric two-qubit wire-order preservation;
- fail-closed behavior when no legal route exists;
- state/determinism validation and a separately reviewed performance baseline.

Batch C does not authorize emitters, provider-specific code generation, remote submission, public API changes, default-path changes, or legacy compiler retirement. Each of those requires a later review gate.

## Requested decision

Approve Batch B exit and Batch C entry with:

`approve IR-PHASE2-BATCH-B-EXIT-BATCH-C`
