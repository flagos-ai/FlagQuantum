# FlagQuantum IR Phase 0 Exit Audit

> Commit references below have been mapped to the publication history.
> Recorded outcomes and approval status are unchanged.

Audit date: 2026-09-01
Audit branch: `codex/open-source-api-convergence`
Baseline commit: `762a9e34b512a8af8e1f14e325ffdfa8b7cde4a3`
Conclusion: **Phase 0 exit gate not passed; Phase 1 implementation remains unauthorized.**

Related material:

- [Phase 0–1 execution plan](../architecture/MULTI_LEVEL_IR_PHASE_0_1_EXECUTION_PLAN.md)
- [Phase 0 factual baseline](IR_PHASE_0_BASELINE.md)
- [Phase 0 corpus design](IR_PHASE_0_CORPUS_DESIGN.md)
- [Metadata inventory](IR_METADATA_INVENTORY.md)
- [Performance baseline and budgets](IR_PHASE_0_PERFORMANCE_BASELINE.md)

## 1. Executive Summary

Phase 0 has produced useful foundations: a public contract inventory, consumer
matrix, six approved ADRs, a 35-opcode manifest, metadata inventory, CPU performance
baseline, and approved Phase 1 internal performance budgets. Focused tests, public
API contract tests, and the default tier are reproducible in the target Docker
environment.

This technical remediation closed `IR0-EXIT-001/002`: each semantic oracle declared
in the manifest is now dispatched and executed per fixture. The corpus contains
12 positive and 14 negative fixtures covering parameter expressions,
complex64/complex128, expectations, two-parameter entangled gradients, measurement
ordering, routing, QASM/QCIS goldens, backend selection, custom 2×2/4×4 matrices,
and dynamic-circuit rejection.

The expanded oracle also exposed and fixed a real legacy defect: `fq.run` previously
overrode `CircuitIR(dtype="complex128")` with the framework's default complex64.
CircuitIR dtype now participates in options resolution as a program constraint,
protected by unit tests and an end-to-end expectation oracle.

Only governance gates remain: final compiler/runtime/training owner reviews are
not recorded, and the API owner has not explicitly authorized Phase 1. Technical
gates are complete, but formal Phase 0 exit remains unapproved.

## 2. Reproduction Environment

```text
container image: flagquantum-dev:pr-check
platform: Linux 6.12.76 linuxkit aarch64
Python: 3.12.13
Torch: 2.13.0+cpu
dtype: complex64
Torch threads: 1
```

These CPU results support no GPU, distributed, or QPU capability claims.

## 3. Machine Verification Results

### 3.1 Phase 0, Documentation, and Public Contracts

Command:

```bash
python -m pytest -q \
  tests/internal_ir/test_phase0_corpus.py \
  tests/unit/test_markdown_code_alignment.py \
  tests/unit/test_public_api_snapshot.py \
  tests/unit/test_ir_public_api.py
```

The original audit reported `40 passed in 2.88s`. After closing the technical
blockers, the Phase 0 corpus file grew to `52 passed`. The final combined review,
including ExecutionOptions precision contracts, documentation, public API
snapshots, and CircuitIR contracts, reported `88 passed in 2.49s`.

Within these results:

- Phase 0 corpus: 52 passed.
- Documentation, public API snapshots, and CircuitIR contracts: 20 passed.
- No drift was found in `CircuitIR` 1.0, stable root exports, or protected contracts.

### 3.2 Default Verification Tier

Command:

```bash
python tools/ci_tier.py pr-default
```

After technical remediation: `1019 passed, 10 skipped, 1268 deselected` in 29.45
seconds. One known PyTorch complex-module warning is not an IR Phase 0 failure.

Runtime integration tier: `168 passed, 31 skipped, 2098 deselected` in 95.88 seconds.

### 3.3 Performance Baseline Reproduction

Rerun with the original parameters:

```bash
python benchmarks/internal/ir_phase0_baseline.py \
  --gate-counts 10 100 1000 10000 \
  --iterations 15 \
  --warmup 3 \
  --run-max-gates 1000
```

Key results:

| Gates | Original plan p95 | Rerun plan p95 | Original plan peak | Rerun plan peak |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.190 ms | 0.316 ms | 11,511 B | 11,511 B |
| 100 | 0.850 ms | 0.812 ms | 70,456 B | 70,456 B |
| 1,000 | 6.876 ms | 6.777 ms | 628,493 B | 628,493 B |
| 10,000 | 67.788 ms | 67.534 ms | 6,288,342 B | 6,288,342 B |

The absolute 10-gate duration is small; fixed-overhead variation does not justify
recalculating budgets. Planning results for 100–10K gates and all peak-memory
values reproduced. The rerun 1K legacy run p95 was 94.36 ms versus the original
64.47 ms. Legacy run throughput is outside the Phase 1 importer+verifier budget,
and this rerun does not replace the approved original baseline.

## 4. P0-001–P0-006 Audit Matrix

| Work package | Status | Existing evidence | Outstanding items |
| --- | --- | --- | --- |
| P0-001 Public contract protection | Technical pass; sign-off pending | Signatures, schemas, hashes, protected surfaces, snapshot tests | Final API-owner baseline review not recorded |
| P0-002 Consumer matrix | Technical pass; sign-off pending | Component matrix, 55-key inventory, typed destinations, drift tests | Final compiler/runtime owner reviews not recorded |
| P0-003 Characterization corpus | Technical pass; sign-off pending | 35 canonical opcodes, 12 positive/14 negative fixtures, serialization, registry drift, parameters/dtypes/custom matrices | Final compiler/runtime/training owner reviews not recorded |
| P0-004 Differential oracles | Technical pass; sign-off pending | Manifest dispatcher executes 12 semantic oracle cases covering state, expectations, gradients, measurements, routing, emitters, backends, and dynamic rejection | Final owner sign-off on oracle coverage and tolerances not recorded |
| P0-005 Performance baseline | Passed | 10/100/1K/10K gates, 15 iterations, 3 warmups, time and memory; reproduced in this audit | No Phase 0 blocker; environment changes must not overwrite original results |
| P0-006 Phase 1 performance budgets | Passed | API-owner approval of internal importer+verifier budgets; derivations protected by tests | Post-implementation budget gates belong to Phase 1 exit |

## 5. ADR Audit

| ADR | Status | Phase 1 constraint |
| --- | --- | --- |
| IR-001 | Approved | Separate program semantics from execution requests |
| IR-002 | Approved | Use a linear value model for qubits |
| IR-003 | Approved | Separate program, compilation, and execution identities |
| IR-004 | Approved | Restrict custom matrices to a verifiable supported subset |
| IR-005 | Approved | Preserve parameter identity, tensor dtype, and late binding |
| IR-006 | Approved | Commit only to the current static CircuitIR reversible subset |

All six ADRs are approved. Approval alone does not complete implementation or exit evidence.

## 6. Formal Phase 0 Exit Gate

| Exit condition | Result | Explanation |
| --- | --- | --- |
| P0-001–P0-006 complete | Technical pass; governance not passed | Technical evidence complete; owner reviews outstanding |
| IR-001–IR-006 approved | Passed | All six marked Approved |
| Corpus covers every static canonical opcode | Passed for opcode coverage only | 35/35; opcode coverage alone does not establish execution of every declared oracle |
| Legacy semantics and performance baseline reproducible | Passed | Semantic dispatcher, runtime/default tiers, and original-parameter performance baseline reproduced |
| Public API and serialization contracts unchanged | Passed | Focused contracts and pr-default passed |
| Blockers, owners, target dates, and rollback recorded | Not passed | Technical blockers closed; formal owner conclusions not recorded |
| Explicit API-owner authorization for Phase 1 | Not passed | Evidence still has `phase1_authorized=false` |

## 7. Blockers

### IR0-EXIT-001: Execute Every Manifest Oracle (Closed)

Parameterized tests dispatch from manifest oracle declarations; unknown oracles
fail immediately. All declarations in the 12 positive fixtures now execute, so
the manifest no longer overstates evidence.

### IR0-EXIT-002: Complete the Minimum Semantic Corpus (Closed)

Added independent/repeated Parameters; add/mul/nested ParameterExpressions;
complex64/complex128 expectations; analytic two-parameter entangled gradients;
measurement ordering; routing legality; QASM/QCIS hashes; custom 2×2/4×4 matrices;
backend selection; and 14 machine-readable negative fixtures. Channels remain
schema characterization. The stable dynamic path rejects requests through
structured `CapabilityError` failures.

### IR0-EXIT-003: Complete Owner Reviews

Closure requires:

- API-owner review of the public factual baseline.
- Compiler/runtime owner review of the consumer matrix, metadata classification,
  and corpus.
- Training owner review of parameter identity and gradient oracles.
- Recorded conclusions in evidence, not chat history or implied agreement.

Closure milestone: after IR0-EXIT-001/002 pass.

### IR0-EXIT-004: Record Explicit Authorization

Closure requires:

- Closure of all preceding blockers.
- Rerunning focused tests, `pr-default`, and performance baseline review.
- API-owner approval of Phase 1 using the authorization packet's exact approval semantics.
- Updating machine evidence without altering public API snapshots or original
  performance budgets to manufacture a pass.

## 8. Rollback

Phase 0 adds only documentation, fixtures, tests, and internal benchmarks. If
Phase 1 does not proceed, these Phase 0-specific assets can be removed. Current
Runtime, Compiler, Stable Core, and public serialized data do not depend on them.

## 9. Final Conclusion

> Phase 0 technical exit evidence is complete and reproducible. Final owner
> sign-off and explicit API-owner authorization remain outstanding, so
> `_compiler` Phase 1 implementation must not begin.

The next audit should focus on owner sign-off and authorization for
`IR0-EXIT-003/004`. It does not need to expand the corpus, overturn the baseline,
or redesign the ADRs.
