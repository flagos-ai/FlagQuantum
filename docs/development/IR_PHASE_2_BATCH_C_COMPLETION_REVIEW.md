# IR Phase 2 Batch C completion review

Status: implementation and approved performance gate complete; owner exit approval pending.

## Delivered

- Deterministic directed coupling graphs and shortest-path selection.
- Explicit logical-to-physical placement with final identity-layout restoration.
- Persistent-layout routing expressed only in the Batch B target gate set.
- Direction-correct CX synthesis without silently exchanging control and target operands.
- Fail-closed behavior for invalid layouts, disconnected topology, unsupported operations, and capacity mismatch.
- State, trainable-gradient, topology-legality, determinism, rerouting, and default-path-zero-impact evidence.
- An approved, executable internal CPU performance gate at 10, 100, 1,000, and 10,000 source gates.

## Approved performance evidence

The 5-iteration, 2-warmup routing-stress validation passed every approved latency, memory, and deterministic-identity threshold.

| Source gates | Emitted gates | Physical SWAPs | p95 / budget | Memory / budget |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 142 | 44 | 3.495 / 5 ms | 320,796 / 1,048,576 B |
| 100 | 652 | 184 | 17.367 / 25 ms | 1,381,180 / 2,097,152 B |
| 1,000 | 5,872 | 1,624 | 202.137 / 500 ms | 12,550,416 / 16,777,216 B |
| 10,000 | 58,072 | 16,024 | 2,943.970 / 3,000 ms | 123,543,476 / 167,772,160 B |

The 10,000-gate latency result has only about 1.9% observed headroom. The gate passed, so this is not an exit blocker, but it is a tracked performance risk. The budget must not be relaxed automatically if later validation fails. These are internal regression limits, not a public SLA.

## Batch D proposed boundary

Batch D should add private, deterministic text emitters for:

- OpenQASM 2;
- a restricted static OpenQASM 3 profile;
- an explicitly versioned QCIS dialect.

Each emitter must consume only verified, routed target-profile IR and must provide exact golden fixtures, parse validation where a parser is available, character-level deterministic output, qubit-order and parameter-format contracts, unsupported-operation diagnostics, and semantic execution fixtures.

Batch D must not add credentials, backend identifiers, provider SDK calls, remote submission, public API exports, default-path integration, or legacy retirement. Format generation is not provider integration.

## Requested decision

Approve Batch C exit and private Batch D emitter implementation with:

`approve IR-PHASE2-BATCH-C-EXIT-BATCH-D`
