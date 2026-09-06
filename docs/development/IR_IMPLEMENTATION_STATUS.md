# FlagQuantum IR implementation status

Updated: 2026-09-06

## Current position

| Phase | Status | Accepted result | Still excluded |
| --- | --- | --- | --- |
| Phase 0 | Complete | Current API/IR/compiler/runtime/deployment semantics and performance baseline characterized | Implementation changes |
| Phase 1 | Complete | Private immutable QuantumIR, verifier, importer, analyses, pass manager, round-trip and semantic differential bridge | Public IR/pass API and default-path use |
| Phase 2 | Complete | Private static canonicalization, target decomposition, directed routing, deterministic emitters, compilation identity/cache and offline end-to-end pipeline | Public/default migration, provider submission and legacy retirement |
| Phase 3 | Private implementation complete | Target capabilities, TargetIR, executable artifacts, runtime adapter, provider-neutral conformance and opt-in shadow comparison | Public/default integration, real provider submission and legacy retirement |
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

Primary verification surfaces:

- `docs/development/IR_PHASE_2_BATCH_F_COMPLETION_REVIEW.md`
- `tests/fixtures/internal_ir/phase2_batch_f_offline_corpus.json`
- `tests/fixtures/internal_ir/phase2_batch_f_performance_baseline.json`
- `tests/fixtures/internal_ir/phase2_batch_f_performance_budget.json`
- `tests/internal_ir/test_phase2_offline_deployment.py`
- `tests/internal_ir/test_phase2_batch_f_performance_budget.py`

## Phase 3 verification surfaces

- `tests/internal_ir/test_phase3_target_capabilities.py`
- `tests/internal_ir/test_phase3_target_legalization.py`
- `tests/internal_ir/test_phase3_executable_artifact.py`
- `tests/internal_ir/test_phase3_runtime_adapter.py`
- `tests/internal_ir/test_phase3_provider_conformance.py`
- `tests/internal_ir/test_phase3_shadow_harness.py`
- `tests/fixtures/internal_ir/phase3_deployment_compatibility.json`

## Next controlled step

Connect the private compiler, executable artifact, runtime adapter, and local CPU
execution path as one minimal vertical slice. Keep public/default integration, real
provider submission, credentials, and legacy retirement outside that slice.
