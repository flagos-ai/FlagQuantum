# Target-CZ block optimization and many-to-many compilation lineage

Status: approved on 2026-11-18; implementation and release qualification pending.
Date: 2026-11-18.
Owner: Integration for the versioned contract; Core for evidence values and
readers; Compiler for transformation and verification; Runtime and Deployment
for read-only handoff adapters.
Classification: `single_device_fast_path`.

## Approved decision

Authorize a bounded, target-dependent block optimization stage in the existing
target compilation workflow, together with physical-plan and compilation-evidence
version `4.0`. It becomes automatic only for the admitted full-CZ profile below.
An accepted rewrite must retain complete many-to-many instruction lineage.
Versions `1.0`, `2.0`, and `3.0` remain readable, with their encodings, identities,
and semantics unchanged. No rewrite is down-converted into a legacy record.

This is separate from the approved native routing default. It does not authorize
native target admission at root `fq.compile`, free final placement, a global-phase
writer, changed artifact envelopes, or a new production dependency. The release
owner must confirm the target release; approval is not release qualification.

## Verified requirement and existing workflow

The existing per-instruction native legalization expresses
`SWAP(0, 1); CX(1, 0)` using four CZ gates in the RZ/RX/CZ basis. The complete
operator needs two CZ gates. A private prototype independently verifies that
reduction, including reversed operands, but cannot enter the target artifact
workflow honestly: `GateDecompositionRecord` links one input instruction to its
replacement, and physical-plan records currently name one source instruction.
Neither record can describe a collective rewrite of several instructions.

The existing CPU/offline user workflow is:

```bash
python -m examples.target_aware_compilation
```

It must keep reporting valid target edges, restored output mapping, and a local
statevector check. The affected artifact workflow is the existing
`compile_circuit_artifact_for_target` entry, exercised by
`tests/integration/test_artifact_compilation_pipeline.py`. These entries exist
today; this proposal does not present a planned root API as available.

The accompanying [operator example](examples/compiler-cz-block-lineage-v4.json)
contains the actual 28-instruction native baseline, a 23-instruction prototype
output, and a proposed aggregate relation between them. Both use RZ/RX/CZ, with
four and two CZ gates respectively. Independent complete-column embedding gives
maximum entry residual `8.163159687020605e-16` after one global phase. This is a
small correctness witness, not a workload-level speedup, a production v4 bundle,
or an implementation of the proposed lineage builder.

## Stage ownership and preserved contracts

Use the existing stage chain:

```text
original source -> topology legalization -> per-instruction native legalization
                -> native block optimization -> final schedule/emission/artifact
```

`legalize_native_gates` and `NativeGateLegalizationResult` retain their current
signatures and meanings. Their program remains the per-instruction native
baseline. Their decomposition records are never relabeled as block records.

Compiler adds a private `NativeBlockOptimizationResult` containing:

```text
source_program             # the exact native legalization output instance
program                    # the accepted optimized CircuitIR
source_content_hash
target_snapshot_id
rewrites                   # typed Core block-rewrite evidence
rejection_counts           # closed counters for rejected optimization proposals
optimization_identity
```

The exact rejection counter keys are `not_lower_cz_cost`,
`synthesis_unavailable`, `residual_rejected`, and
`instruction_growth_rejected`. Ineligible instructions are collection boundaries,
not rejected proposals.

Append `native_block_optimization: NativeBlockOptimizationResult | None = None`
to `TargetLegalizationResult`. Retain existing field order and other fields.
An absent stage preserves the existing native-to-final object chain. A present
stage must retain the exact baseline instance, bind its hash and target, and
own the final CircuitIR before scheduling. The stage identity binds its input,
output, target, accepted rewrites, and rejection counters. Existing identities
and payloads stay unchanged when no rewrite is accepted.

This result has one current responsibility: retaining an actual input and output
for a collective transformation. An interface alone is insufficient. Tests must
replace the block implementation with an identity implementation without
changing the target-compilation, emission, or deployment consumers.

## Initial admitted target profile

- The target verifies full CZ as its only native two-qubit entangler and supports
  parameterized RZ and RX. Other declared one-qubit gates may remain native.
- The verified topology is undirected, or absent. Directed-CX and allocated-idle-
  slot workflows continue through their existing paths; this proposal does not
  redefine their contracts or claim that they support this optimization.
- Blocks contain static numeric, unitary, one- and two-qubit instructions over
  one physical pair. Reversed operands are converted to a common ordered basis.
  Static leading local operations may join a pair block.
- Nonunitaries, symbolic/trainable/batched parameters, custom matrices,
  conditions, classical effects, and unknown annotations form boundaries.
- Routing provenance may be consumed only after validating it against the
  retained topology result. The original routed program and every mapping
  transition remain available. Original metadata is not deleted to unlock a
  rewrite, and synthesized leaves are not given a fictitious single-gate origin.

Collecting a pair across intervening instructions is admitted only when those
instructions are disjoint from its physical support and impose no ordering,
classical, or measurement dependency. Emit a replacement at the first collected
position. Validate that all overtaken instructions commute by this conservative
structural rule; do not add a numerical commutation claim.

The existing target entry signatures stay unchanged. They invoke the new stage
automatically when this profile is satisfied. Other target profiles retain the
existing compilation path. A valid baseline is retained if a proposed rewrite
fails cost, residual, or growth admission; this is rejection of an optimization,
not execution-backend fallback. Report these rejections through the stage's
closed counters. Invalid input and target evidence still fail at the existing
earliest knowable validation stage.

## Cost, equivalence, layout, and resource admission

The native baseline is legalized and budget-checked before block search. Count
its actual full CZ operations. Every emitted opcode and parameter must remain
legal for the same target. A candidate is accepted only when it strictly reduces
CZ count, its 4x4 operator satisfies

```text
U_input = (phase_real + i * phase_imag) * U_output
```

within maximum absolute entry residual `1e-11`, and the phase has unit modulus
within `1e-11`. Recompute this check from the actual operators; a serialized
residual is an attestation, not its own proof. Compiler owns bounded matrix
algebra and synthesis; Core does not import Compiler, Simulation, or NumPy.

Global phase is recorded as an equivalence witness, not applied to IR. The
target-deployment equivalence profile is explicitly up to global phase. Do not
enable this stage in generic `optimize` or advertise its output for coherent
control construction or exact-amplitude preservation. The separately approved
but unimplemented global-phase field proposal does not authorize a writer here.

The original topology transitions are indexed in the retained topology program,
not the optimized output. Replay them there. A fused block may eliminate the
physical gates that separately implemented a SWAP, but must reproduce their
collective physical operator and boundary assignment. Initial, pre-restore,
and final mappings describe those original topology boundaries. A pre-restore
boundary inside a fused block is not claimed to be an intermediate output-gate
boundary. Identity final restoration and logical result ordering remain required.
A logical SWAP cannot be absorbed into an unreported permutation.

Resource bounds:

- At most 64 native input operations per block and at most 32 additional output
  operations relative to that block's native baseline.
- Preserve the existing per-instruction `max_added_operations` admission before
  the new stage. The combined output must also respect the caller's existing
  `max_native_added_operations` budget relative to the routed input. A block may
  consume only remaining headroom. This stage cannot rescue an input that the
  existing baseline legalization already refuses for budget exhaustion.
- Matrix work is bounded at 4x4 per proposal; programs and lineage remain linear
  in retained input/output size. No global plan cache or second circuit IR.
- Serialized v4 bundles reuse existing bundle-byte and record-count limits;
  bounds also cover total nested lineage and rewrite membership entries.
- SDKBench still enforces external process time/memory budgets. More one-qubit
  gates, depth, compile time, or memory are possible and must be reported.

## Version 4 evidence

Keep the compilation-evidence envelope, source/target/output roles, canonical
JSON, duplicate-key rejection, and SHA-256 identity rules. Change the envelope
and physical-plan version to `4.0` only when at least one block rewrite exists.
The initial v4 profile uses the undirected/unallocated v1 physical-plan fields,
replacing `instructions` with the records below and adding exactly:

```text
native_input_circuit_hash
native_input_instructions
block_rewrites
native_block_optimization_identity
mapping_transition_space             # exactly "topology_program"
```

Each dense `native_input_instructions` lineage record has exactly:

```text
instruction_index
source_instruction_index
topology_instruction_index
native_replacement_ordinal
origin
opcode
source_logical_qubits
physical_qubits
```

These records describe the retained native baseline through the existing
per-instruction decomposition map. A routing SWAP's source index is an ancestry
anchor, not a claim that its operator equals that source gate. Mapping transitions
continue identifying the original topology SWAP. No gate parameters, matrices,
target snapshots, or provider state are duplicated in the evidence.

Each block-rewrite record has exactly:

```text
input_instruction_indices            # native-baseline index space
output_instruction_indices           # final program index space
physical_qubits                      # ordered pair
baseline_cz_count
replacement_cz_count
equivalence                          # "unitary_up_to_global_phase"
phase_real
phase_imag
max_abs_residual
```

Each final `instructions` record has exactly:

```text
instruction_index
native_input_instruction_index       # integer or null
block_rewrite_index                  # integer or null
block_replacement_ordinal             # integer or null
opcode
physical_qubits
layer
predecessors
dependency_kinds
```

An unchanged native instruction references exactly one native input index and
has null block fields. A rewritten instruction has a null native input index
and references exactly one block plus its output ordinal. Source ancestry is
resolved through that block's complete input list. Never manufacture one source
index or logical operand tuple for an entire collective rewrite.

Input membership lists are sorted, unique, nonempty, disjoint between blocks,
and at most 64 entries. Output lists are dense within each emitted replacement,
unique, and disjoint; an empty list is permitted for an identity cancellation.
Every native input belongs to exactly one block or one unchanged output. Every
output belongs to exactly one block or one unchanged input. Dense record indices,
known opcodes, actual target edges, pair support, conservative reordering rules,
mapping replay, schedules, and all source/stage identities must be verified.
Rewriting records for an empty output remain in the plan.

Core reading proves closed structure, bounds, continuity, and identity integrity.
It does not prove quantum equivalence from hashes or a claimed residual.
Compiler's verifier requires the actual retained source, topology, native
baseline, optimized program, and target; it recomputes lineage, cost, and block
operators. Runtime and Deployment remain read-only identity/target/result-order
handoff verifiers and never run numerical synthesis. SDKBench independently
checks the emitted complete circuit against its task reference.

## Alternatives considered

Keeping the prototype as an explicit private pass preserves today's contracts,
but leaves ordinary target compilation unable to use the demonstrated reduction.
Mislabeling a collective rewrite as one gate decomposition discards ancestry and
is rejected. Adding the stage before native legalization would change what
`NativeGateLegalizationResult` means; optimizing its retained baseline afterward
keeps that existing responsibility intact. Always importing an external SDK for
compilation would cross the native IR and target ownership boundary and add a
production dependency that this verified requirement does not need.

## Compatibility, rollout, and acceptance

Existing stored bundles and artifact envelopes are not rewritten. No fields are
added to ProgramArtifact v2/v3. New v4 readers accept earlier versions through
the existing explicit dispatcher. Older readers reject v4; stripping block
evidence or flattening it into legacy single-source records is prohibited.
Qualifying target compilations can change instructions, schedule, hashes, and
evidence return type. Document this behavior before release. Nonqualifying or
unimproved programs keep the existing evidence format and identities.

Migration for a consumer that stores compilation evidence:

```text
Before: dispatch evidence versions 1.0 / 2.0 / 3.0.
After: retain those readers and add the strict 4.0 reader before accepting
       artifacts compiled with the admitted automatic block stage.
```

Do not reconstruct a final instruction's single-source ancestry from v4. Resolve
an unchanged instruction through its native input, or a rewritten instruction
through its block's input set. Older consumers must reject v4 explicitly.
No v1-v3 reader is deprecated or removed by this proposal; first deprecation and
removal versions are therefore not applicable. The first release emitting v4
remains unassigned pending the release owner's compatibility review. There is no
claim that current released readers already support this migration. Release notes
must identify the evidence-reader requirement and possible instruction/hash changes.

Required delivery order:

1. Land the approved Integration contract, a contract fake, and conformance tests.
2. Implement Core values/readers with v1-v3 identity fixtures, strict malformed-
   record cases, nested limits, and actual version-dispatch tests.
3. Implement the Compiler stage and verified physical-plan construction behind
   the existing target workflow. Replace the implementation without modifying
   consumers. Prove source hashes and mapping evidence still describe the
   original user program.
4. Extend Runtime and Deployment handoff acceptance in their own changes. Reject
   a v4 bundle paired with stale source, baseline, target, layout, or output.
5. Test reversed pairs, routed SWAP/CX cancellation, interleaved spectators,
   identity cancellation, barriers, symbols/gradients, custom matrices, corrupted
   residuals, missing input membership, and all resource limits. Test real
   workflows, not just field inventories.
6. Run the required default and local integration tiers. Evaluate held-out tasks
   with unchanged SDKBench operator/topology/layout and Linux resource acceptance.
   Amend participant revision registration transparently; do not relax thresholds
   or give any SDK an exclusive layout or budget. Report counts, depth, compile
   time, memory, failures, unsupported cases, and regressions independently.

The exact [candidate contract](../../contracts/compiler-cz-block-lineage-v4-candidate.json) records the admitted fields and bounds.
The user explicitly approved the complete PR #1221 proposal scope on 2026-11-18:
the automatic qualifying-CZ stage, appended result field, and v4 many-to-many
evidence read/write, preserving v1-v3 reading and identity semantics. The approval
excludes root native target admission, free final layout, and global-phase writing.
API-owner approval does not waive CODEOWNER review,
Integration-first delivery, capability evidence, or required checks.

## Integration contract fake

`tests/unit/test_compiler_cz_block_contract_fake.py` retains the actual numerical
baseline and prototype output from the operator example. A single test consumer
accepts either an identity stage or the recorded reducing stage, then uses the
existing dependency scheduler and OpenQASM emitter without changing those
consumers. The reducing fake emits two CZ operations instead of four. Its
complete-column operator oracle embeds explicit RX/RZ/CZ definitions, independently
of Compiler matrix multiplication and synthesis, and recomputes both block and
complete-program equivalence. The numerical fixture remains unchanged.

The fake checks exact baseline instance retention, target and stage identity,
closed rejection counters, exclusive memberships, CZ savings, phase and operator
residuals, empty-output cancellation, and block/combined growth limits. Tests
recompute identities after tampering to distinguish structural or numerical
rejection from a stale checksum. The fake's identity is explicitly test-domain
only; it does not assign a production v4 serialization encoding.

Run the review-only conformance path with:

```bash
python -m pytest tests/unit/test_compiler_cz_block_contract_fake.py -q
```

This is an executable Integration prerequisite, not evidence that a production
v4 bundle can already be read, compiled, or deployed. Actual target-artifact
replacement, routing transition replay, legacy bundle identity fixtures, strict
v4 parsing, and Runtime/Deployment handoff remain the subsequent delivery gates
above. The fake does not optimize arbitrary programs or claim performance parity.

`tests/unit/test_compilation_evidence_legacy_identity.py` freezes three synthetic
pre-v4 bundles from the existing Core test builders at
`c5f212000392df18a667bf6ac164f08bd8f33942`. The fixtures cover an unallocated
undirected/no-topology plan, directed-CX reversal, and allocated workspace with
routing restoration and logical result projection. Tests pin canonical text,
bundle identities, plan identities, explicit dispatch types, and layout/direction
semantics. They also require exact rejection of stale source, target, or output
evidence. These are codec regression fixtures, not hardware or numerical evidence;
do not regenerate them to accommodate v4 implementation changes.
