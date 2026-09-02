# FlagQuantum IR implementation status

Updated: 2026-09-02

## Current position

| Phase | Status | Accepted result | Still excluded |
| --- | --- | --- | --- |
| Phase 0 | Complete | Current API/IR/compiler/runtime/deployment semantics and performance baseline characterized | Implementation changes |
| Phase 1 | Complete | Private immutable QuantumIR, verifier, importer, analyses, pass manager, round-trip and semantic differential bridge | Public IR/pass API and default-path use |
| Phase 2 | Complete | Private static canonicalization, target decomposition, directed routing, deterministic emitters, compilation identity/cache and offline end-to-end pipeline | Public/default migration, provider submission and legacy retirement |
| Phase 3 | Not authorized | TargetIR and executable ABI design is ready for entry review | Any implementation, shadow/default integration or provider work |
| Phase 4 | Not started | ProgramIR and unified dynamic-circuit direction exists in architecture | Implementation |
| Phase 5 | Not started | Timing/QIR/advanced lowering candidates exist in architecture | Implementation and production claims |

## Phase 2 accepted capability

The accepted profile is `private_static_compiler_v1` under `flagquantum._compiler`:

```text
CircuitIR import/seal
  -> static canonicalization
  -> RX/RY/RZ/CX decomposition
  -> directed placement/routing
  -> post-routing canonicalization
  -> deterministic compilation identity/cache
  -> restricted OpenQASM 2 / OpenQASM 3 / QCIS emission
```

It has structural, state, expectation, gradient, order, parser, text-hash,
fail-closed and performance evidence. It is not a public or default compiler.

Primary evidence:

- `contracts/ir-phase2-exit-authorization.json`
- `docs/development/IR_PHASE_2_BATCH_F_COMPLETION_REVIEW.md`
- `contracts/ir-phase2-batch-f-performance-budget-validation.json`
- `tests/fixtures/internal_ir/phase2_batch_f_offline_corpus.json`

## Next controlled step

Phase 3 should establish an internal target and executable boundary before any attempt to
make backend switching transparent. Entry must begin with a provider-free
`TargetCapabilities` model and identity rules. Public API, default-path shadow execution,
real provider dispatch, credentials and legacy retirement require later independent
evidence and approval.

The entry proposal is `docs/development/IR_PHASE_3_ENTRY_APPROVAL_PACKET.md`.
