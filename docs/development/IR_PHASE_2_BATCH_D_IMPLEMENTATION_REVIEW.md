# IR Phase 2 Batch D implementation review

Status: private emitter implementation complete; performance budget approval pending.

## Delivered

Batch D adds three private, deterministic emitters over verified internal `QuantumModule`:

- OpenQASM 2.0 static `RX/RY/RZ/CX` profile;
- restricted-static OpenQASM 3.0 `RX/RY/RZ/CX` profile;
- versioned `flagquantum_qcis_rx_ry_rz_cx_v1` profile lowered to the repository's
  existing QCIS-native `X2P/X2M/Y2P/Y2M/RZ/CZ` instruction vocabulary.

The profile version is carried by the immutable emission result rather than injected as
non-standard text into the QCIS payload. This keeps the emitted QCIS payload compatible
with native consumers while retaining an auditable dialect identity.

Each successful result binds the source program identity, exact text, content SHA-256,
and emitter profile. Unsupported or invalid inputs return structured diagnostics and no
partial text.

## Evidence

The focused suite covers:

- exact golden output for all three formats;
- parser fixtures and statevector semantic equivalence;
- asymmetric control/target order preservation;
- canonical finite-number formatting and character-level determinism;
- stable content hashes and source-program binding;
- fail-closed symbolic, trainable runtime-binding, unsupported-op, malformed-module,
  and non-module cases;
- proof that no emitter is exported from public `flagquantum` and the default path is
  unchanged.

The emitters intentionally accept only verified `quantum.rx`, `quantum.ry`,
`quantum.rz`, and `quantum.cx` modules. Target decomposition and placement/routing remain
separate earlier pipeline responsibilities. Trainable parameters must be resolved into a
static deployment artifact before text emission; Batch D does not silently read a runtime
binding table or stringify symbolic expressions.

## Performance baseline and proposed budget

Four five-iteration measurements were retained, including scheduler-tail outliers. Two
runs used the final QCIS payload form. The benchmark measures emission from a prebuilt,
verified target module, so importer, decomposition, and routing costs are not attributed
to the emitter.

The worst observed p95 latency across the retained runs was:

| Gates | OpenQASM 2 | OpenQASM 3 | QCIS v1 |
| ---: | ---: | ---: | ---: |
| 10 | 0.095 ms | 0.087 ms | 0.090 ms |
| 100 | 0.398 ms | 0.389 ms | 0.695 ms |
| 1,000 | 12.780 ms | 5.143 ms | 3.544 ms |
| 10,000 | 81.856 ms | 98.801 ms | 62.607 ms |

Peak traced memory at 10,000 gates was approximately 3.16 MiB for OpenQASM 2,
3.17 MiB for OpenQASM 3, and 4.04 MiB for QCIS v1. Every repeated output hash was
deterministic.

The proposed internal budget preserves at least 25% headroom over every retained latency
and memory maximum. It is deliberately unapproved and is not a public SLA. No test or
benchmark may treat it as an active gate until owner approval.

## Explicitly not delivered

Batch D does not add or authorize:

- provider SDK calls, remote submission, credentials, or backend IDs;
- public emitter APIs or changes to `fq.run`, `fq.plan`, or default compilation;
- dynamic control flow, measurement/classical registers, pulse/timing programs, or
  symbolic parameter emission;
- replacement or retirement of the legacy QASM/QCIS exporters;
- Batch D exit, Batch E provider packaging, or Phase 2 exit.

Format generation is a compiler boundary, not proof of quantum-cloud integration.

## Requested decision

Approve only the proposed private Batch D performance budget and its machine-enforced
gate with:

`approve IR-PHASE2-BATCH-D-PERFORMANCE-BUDGET`

That approval must keep provider work, public/default-path changes, legacy retirement,
Batch D exit, Batch E, and Phase 2 exit closed.
