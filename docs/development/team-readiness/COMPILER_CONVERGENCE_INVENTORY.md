# Compiler convergence inventory

Status: stable Compiler authority closed; protected product and private-candidate convergence remain

Updated: 2026-09-06

Path classification: `single_device_fast_path` and provider-free static compilation

Public API impact: none

## Decision

Stable program transformation now has one authoritative package. Runtime owns
planning and plan assembly. The remaining migration work is limited to protected
cross-domain plan products and the separately gated private compiler candidate.

- `flagquantum.compiler` is the factual authority for stable canonical
  optimization, instruction scheduling, backend lowering, coupling maps, and
  topology routing. The former public forwarding file and the corresponding
  `compilation.compiler`/`compilation.routing` modules were removed.
- `flagquantum.runtime.planner` now owns stable/default execution selection,
  estimates, backend calibration, planning orchestration, and plan assembly. The
  transitional `flagquantum.compilation` package retains only execution-plan
  products, serialization/validation/reconstruction helpers, and the compatibility
  calibration seam until shared products gain Core-owned contracts.
- `flagquantum.compiler.noise` now owns stable noise-model lowering. The former
  `flagquantum.compilation.noise` package was deleted; noisy execution-plan
  models remain with the transitional plan product rather than Compiler.
- `flagquantum._compiler` is the factual authority for the private
  `private_static_compiler_v1` candidate: exact import, immutable internal IR,
  verifier, analyses, pass manager, static lowering, deterministic text emission,
  compilation identity/cache, TargetIR legalization, and sealed executable
  artifacts. Its package `__init__` intentionally exports nothing and the default
  path does not import it.
- The default CPU path now calls `flagquantum.compiler` directly for program
  transformation and uses `flagquantum.compilation` only for the protected plan
  product and its contract helpers.
  `_compiler` remains off the default path and must replace proven concerns rather
  than being copied wholesale into the stable package.

Consequently, `_compiler` is not a replacement for all of `compilation` today.
It is the best implementation candidate for a narrowly defined static program
transformation slice. Runtime planning and backend selection currently mixed into
`compilation` belong behind Runtime-owned policy contracts, while cross-domain IR,
capability, request, and artifact types require a Core-owned contract first.

## Current entry-point inventory

“Authority” below means current factual ownership, not the desired final directory.

| Concern | Current entry points | Input → output | Current authority and notes |
| --- | --- | --- | --- |
| Program capture | `Circuit.to_ir()`; `CircuitIR`; `Module._compile_builder_program()` | Python circuit/builder → `CircuitIR` or Runtime builder template | Public circuit capture is split between Core/API and Runtime. Neither compiler tree owns a unified capture boundary. Runtime builder capture is a misplaced compiler-like responsibility but is outside this team's edit scope. |
| Normalization/import | `compiler.pipeline._as_ir()` through `ensure_circuit_ir()`; `_compiler.importers.circuit_ir.import_circuit_ir()`; `_compiler.exporters.circuit_ir.seal_circuit_ir_round_trip()` | `Circuit`/`CircuitIR` → `CircuitIR`, or validated `CircuitIR` → `ImportedCircuitProgram`/`SealedCircuitIRRoundTrip` | `compiler` performs stable public normalization. `_compiler` owns the exact, allowlisted private import profile and structured diagnostics. |
| Validation | `CircuitIR.validate()`; stable checks in `runtime.planner.plan()`; `_compiler.ir.verifier.verify_module()`; `TargetCapabilities`/`TargetIR` constructors and `legalize_quantum_module()` | candidate program/target → accepted value or typed failure | Core validates public IR. `_compiler` has the only compiler-internal verifier. Stable execution-request validation belongs to Runtime planning. |
| Analysis | `runtime.planner.analyze()`; `_compiler.analyses.AnalysisManager`, `DefUseAnalysis`, `QubitLifetimeAnalysis` | IR/module → structural or reusable analysis result | Runtime structural analysis feeds execution planning; `_compiler` analyses feed private passes and are not default-path consumers. |
| Pass execution | `compiler.simple_compile()` and its three functions; `_compiler.passes.PassManager`; `_compiler.passes.static_canonicalization`; target decomposition and placement/routing passes | program IR → transformed program IR | Stable canonicalization is authoritative in `compiler`; `_compiler` remains a private replacement candidate with descriptors, diagnostics, analyses, and pipeline identity. |
| Routing | `compiler.routing.route_to_topology()` and `select_routing_strategy()`; `_compiler.passes.placement_routing.PlacementRoutingPass` | logical program + coupling graph → routed program | Stable routing is authoritative in `compiler`; the private implementation has a different graph and evidence model and remains off the default path. |
| Lowering | `compiler.compile_for_backend()`; `compiler.lower_noise_model()`; `_compiler.passes.DecomposeToTargetGateSetPass`; `_compiler.target_legalization.legalize_quantum_module()`; test-only `lower_module_for_differential()` | source/logical IR + target/noise constraints → lowered IR or `TargetIR` | `compiler` owns stable local, topology, and noise lowering; noisy execution-plan products remain transitional in `compilation`; `_compiler` owns private TargetIR lowering. |
| Backend/mode selection | `runtime.planner.select_backend_by_cost()`; `select_execution_mode()`; `plan_runtime_selection()`; noisy-backend selection | `CircuitIR` + execution/resource policy → backend/mode candidate or selection plan | Runtime is authoritative for execution selection. `_compiler.TargetCapabilities` describes target legality and does not select a runtime backend. |
| Scheduling/planning | `compiler.schedule_layers()`; `runtime.planner.plan()`, `plan_advanced()`, `plan_for_backend()` | compiled IR + resolved execution options → `ExecutionPlan` | Compiler owns instruction scheduling; Runtime owns orchestration and final plan assembly; the protected plan product remains transitional. |
| Code generation | `_compiler.exporters.text.emit_openqasm2()`, `emit_openqasm3()`, `emit_qcis_v1()`; legacy `flagquantum.utils` exporters used by deployment | verified static module → canonical text + content hash | `_compiler` owns the deterministic private emitters. `compilation` has no generic code-generation boundary. Legacy utility emitters remain separate consumers and are not retired here. |
| Artifact packaging | `compilation.execution_plan_contract.attach_execution_contract()` and plan serialization; `_compiler.exporters.circuit_ir`; `_compiler.offline_deployment.compile_offline_static()`; `_compiler.executable_artifact.seal_executable_artifact()` | compiled program/plan/bytes → identity-bound plan, round-trip envelope, offline result, or executable artifact | Both trees define cross-stage products. Core already contains a protected candidate `ProgramArtifact`, but neither compiler path uses it. Cross-domain convergence therefore needs a Core contract before implementation migration. |

## Responsibility overlap and dependency findings

### Duplicated responsibilities

| Overlap | `compilation` implementation | `_compiler` implementation | Material difference |
| --- | --- | --- | --- |
| Canonicalization | `remove_identity_gates`, `merge_self_inverse`, `merge_adjacent_rotations`, `simple_compile` | the three corresponding passes and `StaticCanonicalizationPass` | The private path has explicit pass contracts, revision/identity checks, diagnostics, and a pipeline digest. |
| Routing/topology | `CouplingMap`, routing strategies, metadata-producing functions | `DirectedCouplingGraph`, `PlacementRoutingPass` | Models and evidence schemas differ; neither delegates to the other. |
| Structural inspection | `analyze`, `schedule_layers` | def-use/lifetime analyses plus IR walkers | The analyses are not equivalent, but both trees own compiler analysis concepts. |
| Lowering | backend compile and noise lowering | gate-set decomposition and target legalization | The legacy path lowers for execution policy; the private path lowers against an explicit static target contract. |
| Identity/product models | `ExecutionPlan` fingerprints and serialization | source/internal/program/pipeline/target/artifact identities | Both build identity chains, but at different lifecycle stages and without a shared Core artifact contract. |
| Result/status models | planning/selection dataclasses and exceptions | typed import/pass/emission/legalization/artifact result objects | Failure vocabulary is fragmented; `_compiler` generally returns diagnostics while stable planning raises public exceptions. |

### Direct package dependency direction

```text
Runtime consumers ───────────────► compiler ──────────────────► Core
                └───────────────► Runtime planning ──────────► Core
                                      _compiler ───────────────► Core
                                      (not on the default path)

compiler    ── no import ──► _compiler
_compiler   ── no import ──► compiler
Runtime     ── no import ──► _compiler
```

The stable and private compiler implementations both depend on Core and do not
import each other. Runtime planning imports the stable compiler, while the private
path remains isolated from default execution.

There are, however, architectural reverse dependencies outside that pair:

- `runtime.planner` owns option resolution and distributed backend policy and
  calls Compiler transformations directly. It still imports transitional
  `compilation` plan models, assembly, and contracts; removing
  that one-way dependency requires moving shared plan products to Core.
- `_compiler.deployment_compatibility` and `_compiler.deployment_dry_run` import
  Deployment modules. These are explicitly authorized bridge/evidence paths, but
  they are reverse dependencies relative to the target Compiler → Core-only rule.
- `_compiler.runtime_abi`, `runtime_adapters`, shadow harnesses, provider
  conformance, and sandbox connector/observation types do not import the real
  Runtime package, but they own Runtime/Provider lifecycle responsibilities inside
  the Compiler tree. They must be split during convergence rather than promoted as
  compiler APIs.

## Runtime dependency classification

Runtime's imports fall into two different categories. Calls through the public
`flagquantum.compiler` facade are intentional service dependencies: Runtime supplies
an input program and consumes a transformed program without owning the algorithm.
They are not evidence that planning still belongs to Compiler. Imports from
`flagquantum.compilation` are the remaining protected product seam and cannot move
until the corresponding Core contract is approved.

| Runtime consumer | Dependency | Classification |
| --- | --- | --- |
| `runtime/execution.py`, `runtime/planner`, `runtime/noise_registry.py` | stable compile and noise-lowering facade calls | intentional Compiler service calls |
| `runtime/backends/statevector/planning.py` | stable layer scheduling facade | intentional service call retained by the raw-program compatibility entry point |
| `runtime/backends/statevector/noisy.py` | stable noise-lowering facade | intentional service call retained by the raw-program compatibility entry point |
| `runtime/dynamic/routing.py` | stable coupling-map and routing facade | intentional Compiler service call |
| `runtime/planner` | `ExecutionPlan` products plus layer-building and contract attachment helpers | protected product seam; plan policy and assembly are Runtime-owned |
| `runtime/execution.py`, `runtime/plan_execution.py`, `runtime/result.py`, `runtime/noise_registry.py` | `ExecutionPlan`, noisy-plan products, validation, decoding, and program reconstruction | protected product/schema seam pending a Core contract |

No Runtime file directly imports `flagquantum._compiler`. That is an important
preserved invariant and must remain true until the replacement contract is approved.

Additional non-Runtime consumers that constrain compatibility are the public API and
Circuit facades, `flagquantum.compiler`, Deployment routing/cloud code, Simulation
noise execution, and Agent Services runtime-selection explanation.

## Current default compilation flow

The table distinguishes program transformation from Runtime policy even though both
currently live under `compilation`.

| Stage | Entry point | Input artifact | Output artifact | Failure behavior | Evidence retained |
| --- | --- | --- | --- | --- | --- |
| 1. Capture | `Circuit.to_ir()` or direct `CircuitIR` construction | user circuit/builder | validated `CircuitIR` 1.0 | Core validation/type errors | canonical JSON and `CircuitIR.content_hash` |
| 2. Stable request validation | `runtime.planner.plan()` | `Circuit`/`CircuitIR`, `ExecutionOptions`, measurements, noise | resolved source IR and options | `TypeError`, `ValidationError`, or `CapabilityError`; conflicts and dynamic circuits fail closed | requested/resolved options later enter plan fingerprints |
| 3. Local compile | `plan_advanced()` → `compile_for_backend()` | source `CircuitIR`, RuntimeConfig, optional coupling map | optimized/routed `CircuitIR` | `CompilationError`, routing/value errors | runtime-config manifest and routing metadata |
| 4. Canonical optimization | `simple_compile()` | `CircuitIR` | fixed-point optimized `CircuitIR` | bounded rounds; raises `CompilationError` if no fixed point | transformed instruction sequence; no standalone pass evidence |
| 5. Optional routing | `route_to_topology()` | optimized IR + `CouplingMap` | routed IR | invalid topology/unsupported routing raises | routing and optional strategy-selection metadata |
| 6. Optional noise lowering | `lower_noise_model()` | compiled IR + `NoiseModel` | channel-bearing `CircuitIR` | validation/capability errors | channel instructions and noisy-plan identity |
| 7. Analysis | `analyze()` | compiled/lowered IR | `CircuitAnalysis` | assumes valid IR; malformed values surface normal exceptions | counts, depth, wire use, noise flags |
| 8. Backend/mode policy | `select_execution_mode()` and cost/policy helpers | source/analysis + resource and accuracy policy | selected state mode and recommendation | unsupported capability or invalid policy fails | candidate reasons, warnings, cost and distribution metadata where applicable |
| 9. Plan assembly | `build_execution_plan()` | compiled IR, analysis, resource estimate, selection | `ExecutionPlan` | constructor/type failure | layers, estimates, routing plan, runtime manifest |
| 10. Contract attachment | `attach_execution_contract()` | plan + source + requested/resolved options + environment | identity-bound serialized `ExecutionPlan` | contract validation/serialization failure | program/options/environment/compiler fingerprints and plan identity |
| 11. Runtime handoff | `runtime.plan_execution.execute_plan()` | validated `ExecutionPlan` | execution input/result | stale or incompatible plan fails closed | actual execution and fallback evidence belongs to Runtime result |

## Existing private static compilation flow

This is the representative line frozen by `tests/team/compiler/`. It remains
explicit, provider-free, and non-default.

| Stage | Entry point | Input artifact | Output artifact | Failure behavior | Evidence retained |
| --- | --- | --- | --- | --- | --- |
| 1. Static preflight | `compile_offline_static()` | object + `OfflineStaticTarget` | accepted static `CircuitIR` | non-IR is `INVALID_INPUT`; requests, dynamics, width mismatch are `UNSUPPORTED_WITH_DIAGNOSTICS` | structured diagnostic; no partial artifact |
| 2. Exact import/seal | `seal_circuit_ir_round_trip()` → `import_circuit_ir()` | validated `CircuitIR` | `ImportedCircuitProgram` + `SealedCircuitIRRoundTrip` | unknown/dynamic/lossy semantics fail closed | source hash, canonical source JSON, internal program identity, bindings, constraints, provenance |
| 3. Internal verification | `verify_module()` | immutable `QuantumModule` | `VerificationResult` | typed diagnostics, no repair | locations and diagnostic codes |
| 4. Pipeline identity/cache | `CachedPipelineRunner` | module + target/topology/calibration/options + pass manager | `CompilationIdentity` and cache disposition | unsafe identity bypasses visibly; failed compilation is not cached | canonical identity JSON, source/input/pipeline digests, cache counters |
| 5. Canonicalize | `StaticCanonicalizationPass` | verified module | transformed module | pass diagnostic stops pipeline | pass descriptor/result, revision, program identity, pipeline digest |
| 6. Gate-set lowering | `DecomposeToTargetGateSetPass` | canonical module | RX/RY/RZ/CX module | unsupported rewrite fails with diagnostics | pass result and derived program identity |
| 7. Placement/routing | `PlacementRoutingPass` | lowered module + directed graph/layout | physical module | invalid graph/layout or unroutable operation fails closed | topology identity, pass result, operation provenance |
| 8. Post-route canonicalize | `StaticCanonicalizationPass` | routed module | final verified module | diagnostic stops pipeline | final module identity and complete pipeline digest |
| 9. Code generation | exact text emitters | final module | OpenQASM 2/3 or QCIS text result | unsupported constructs return diagnostics; failed emission is not reported as compiled | format, canonical text, SHA-256 content hash |
| 10. Result packaging | `OfflineCompilationResult` | source envelope + pipeline execution + emissions | immutable private result | any failed emission makes the whole result unsupported | source, compilation, module, cache, diagnostic, and emission identities |

The later private TargetIR/artifact path is separate:
`legalize_quantum_module()` produces a capability-bound `TargetIR`, an approved
artifact-profile encoder produces canonical bytes, and
`seal_executable_artifact()` binds source, target, capability, compilation, payload,
profile, and artifact identities. It is not used by the default Runtime.

## First minimal replaceable slice

The first candidate is **static local canonical optimization only**:

```text
CircuitIR
  -> exact private import/seal
  -> Phase 2 Batch A static canonicalization passes
  -> verified lowering back to CircuitIR
  -> CircuitIR
```

The eventual replacement adapter would sit behind the existing
`compiler.simple_compile(circuit_or_ir) -> CircuitIR` boundary. Runtime,
`compile_for_backend`, public imports, and callers would remain unchanged. This is
smaller and safer than first replacing routing, planning, backend selection, noise,
or artifact serialization.

The physical authority move does **not** switch to the private implementation.
Before a later implementation replacement,
the integration branch must approve a replacement contract and prove:

1. the accepted-input domain matches the current `simple_compile` domain, including
   metadata and trainable parameter cases, or unsupported inputs have an explicitly
   approved failure contract;
2. instruction semantics, state, expectation, gradients, dtype/device, request
   ordering, and exception behavior match;
3. output and pipeline identities are deterministic across processes;
4. a legacy implementation and the candidate both pass the same conformance suite;
5. Runtime and public API consumers require no source changes;
6. there is no silent legacy fallback. Any temporary compatibility adapter has an
   owner, scope, removal condition, and target version.

The new team characterization tests cover deterministic output, semantic parity with
the current optimizer, invalid/unsupported fail-closed behavior without partial
artifacts, preservation of classified program and instruction metadata, and the
source/pipeline/target/emission identity chain. They also record the remaining input
domain gap: `simple_compile` preserves arbitrary top-level metadata, while the
private importer rejects unclassified metadata rather than silently dropping it.
These tests are evidence for the candidate slice, not authorization to switch it on.

## Human-maintainability notes for the next slice

**Primary domain:** Compiler. The first replacement candidate remains the static
canonicalization path behind `compiler.simple_compile`; the physical directory
migration does not authorize changing semantics or the public API.

**Readable scenario:**
`tests/team/compiler/test_static_pipeline_characterization.py` is the existing
ten-minute path. It demonstrates deterministic cache behavior, semantic
equivalence with `simple_compile`, invalid/unsupported input failure without
partial artifacts, classified metadata preservation, the explicit unclassified
metadata domain gap, and source/pipeline/target/emission identity binding.

No new legality contract is retained in this round. Existing target capability
coverage stays expressed by `CompilerRequirementProjection.compare_available()`
and its authoritative `CapabilityComparison` result. This keeps the inventory
focused on the current Compiler behavior and the smallest CPU migration slice.

## Core contract proposal

No protected Core file is changed by this work. The Core/integration teams should
handle these types through a versioned proposal, compatibility analysis, fake, and
conformance tests before Compiler migration begins.

| Current Compiler-owned type(s) | Proposed Core authority | Reason and boundary |
| --- | --- | --- |
| `_compiler.ir`: `IRType`, `ValueId`, `ValueRef`, `Operation`, `Block`, `Region`, `QuantumModule`, schema descriptors | versioned Core logical/quantum IR payload contracts | These encode program semantics and identity. Compiler should construct and transform them, not own the cross-stage semantic schema. Avoid creating a second public IR beside `CircuitIR`; reconcile them under an approved versioned artifact design. |
| `TargetCapabilities`, `GateCapability`, `ParameterConstraint`, target/artifact/control/measurement enums | Core capabilities contract | Capability vocabulary is consumed across Compiler, Runtime, and Providers and must be vendor-neutral. |
| `TargetIR` and `TargetOperation` | Core physical/executable program artifact payload | Once handed to Runtime/Provider, target-program schema and identity are cross-domain. Legalization remains a Compiler implementation. |
| `SealedCircuitIRRoundTrip`, `SealedExecutableArtifact`, `ArtifactProfile`, artifact identity fields | Core `ProgramArtifact`/executable artifact contract | Core already has a protected candidate `ProgramArtifact`; integration should extend or adapt that authority rather than duplicate another envelope. Sealing/code generation remain Compiler operations. |
| `SourceIdentity`, `SourceProvenance`, `InstructionSemantics`, `CompilationIdentity` identity/evidence payload | Core provenance and compile-evidence contracts | Serialized identities that Runtime or Provider verifies must have one schema owner. Digest computation remains Compiler behavior. Pure cache counters can remain Compiler-private. |
| `InternalExecutionRequest`, observable/measurement request records, `ImportConstraints` | Core execution-request and program-requirements contracts | Requests and numerical constraints cross Compiler/Runtime boundaries. They must be reconciled with existing Core/Public measurement and options semantics without changing stable behavior. |
| `SymbolicParameter`, `SymbolicExpression`, `RuntimeBindingRef`, `BindingTable` | Core parameter/binding contract, reconciled with `core.parameters` | Parameter meaning and late binding must survive compilation and execution without framework-specific objects leaking across the boundary. |
| `Diagnostic` location/severity and cross-domain failure codes | Core diagnostic/failure envelope; compiler-only codes remain Compiler extensions | Runtime/Provider-visible failure evidence needs one versioned shape. Pass-internal diagnostics need not become public. |
| `compilation.models.ExecutionPlan` and its serialized nested contract | Core execution-plan contract | It is already a protected stable object and Runtime imports it directly from Compiler. Move only through the approved API proposal with compatibility imports and serialization fixtures. |

Types that should **not** move wholesale to Core:

- pass implementations, `PassManager`, analysis caches, compiler pipeline options,
  cache eviction state, emitter implementations, and compilation algorithms remain
  Compiler-owned;
- backend cost selection, runtime-mode policy, resource orchestration, and candidate
  ranking should move from `compilation` toward Runtime-owned implementations while
  consuming Core contracts;
- adapter lifecycle, handles, submission receipts, status/cancel/fetch behavior,
  shadow execution, provider conformance drivers, and sandbox transports currently
  under `_compiler` should move to Runtime or Provider domains. Only their stable
  request/result/identity data belongs in Core.

## Exit conditions and blockers

The **stable Compiler authority move is complete**: transformation, scheduling,
routing, and noise lowering have one facade; Runtime owns selection and plan
assembly; and the CPU path executes a restored plan without recompiling it.

The broader machine-readable `compiler_convergence` track correctly remains
`in_progress`. It is not complete until all of the following are true:

- Core contracts land first and both compiler implementations can consume them;
- the static optimization slice passes shared replacement/conformance tests;
- Runtime imports only a Core artifact/plan contract and the narrow Compiler facade,
  not Compiler implementation modules or Compiler-owned product schemas;
- Compiler no longer imports Runtime or Deployment implementations;
- all supported compiler entry points delegate to the target facade;
- legacy internal imports reach zero and the old implementation is deleted or
  explicitly time-bounded by a compatibility record;
- public signatures, defaults, exceptions, schemas, and `fq.plan`/`fq.run` behavior
  remain protected throughout migration.

Core Target Capabilities v1 and its loss-accounted Compiler adapter now exist, but
the richer Compiler target fields still require the legacy comparator. Current
blockers are the incomplete executable artifact/request contracts, the stable
`ExecutionPlan` definition living in `compilation`, and incomplete accepted-domain
equivalence between `simple_compile` and the stricter private importer. Intentional
Runtime calls through the stable Compiler facade are not blockers and must not be
removed merely to reduce an import count.
