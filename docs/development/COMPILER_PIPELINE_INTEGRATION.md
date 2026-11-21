# Native compiler integration candidate

This development record combines four compatible implementations behind the
existing compiler and IR contracts. The source candidate is
`4cfa415bd720c6d1f42ab27de70102c17c10e880`; its control is main
`2d78133a1f5a7d234acbb128724ccf2674e2eeb8`. Both are complete source trees,
loaded in separate Python processes. The classification is
`single_device_fast_path`, with no scalability or capability promotion.

## Included changes

- [Two-qubit fallback reuse](COMPILER_TWO_QUBIT_FALLBACK.md) avoids solving the
  same rejected full block twice, retaining narrower fallback opportunities.
- [Classical reader ordering](COMPILER_CLASSICAL_READ_ORDER.md) makes a classical
  overwrite depend on every earlier reader of the value, preserving feedback
  semantics when ready quantum operations move independently.
- [Native hash reuse](COMPILER_NATIVE_HASH_REUSE.md) reuses hashes within one
  result construction; independent constructors and verifiers still hash their
  inputs. It introduces no lifetime or global cache for mutable tensors.
- [IR scalar encoding](IR_SCALAR_ENCODING.md) bypasses protocol inspection only
  for exact built-in scalar types, retaining subclass, tensor and symbolic
  dispatch and canonical encoding.

These are the source changes reviewed separately in PRs #1264–#1267. Combining
their commits allows the actual delivered pipeline to be tested before one
protected integration merge. Public exports, signatures, defaults, schemas,
native block acceptance and output-layout contracts are unchanged. In particular,
the [equal-cost depth experiment](COMPILER_SABRE_DEPTH.md) remains confined to an
explicit diagnostic and is not enabled in production.

## Combined independent acceptance

[Baseline observations](evidence/compiler-pipeline-baseline.json) and
[candidate observations](evidence/compiler-pipeline-candidate.json) each retain
120 complete application outcomes. SDKBench is unchanged at
`9aa9260920984e2ca05bcb293abb885fccd9c144`. Inputs cover five application
families, widths 6/8/12, four topologies, and ordinary-restored plus target-artifact
workflows, using the inspected public nonce `01` repeated 32 times.

Every outcome passes the existing independent evaluator, including target edges,
normalization, compile caps and identity output placement. Every paired
normalized operation/layout digest and every evaluator metric match exactly.
Six-qubit acceptance uses complete operators up to a global phase; widths 8 and
12 use three probes. This is not universal operator proof at the larger widths.
Full serialized artifacts contain clock-dependent target identities and are not
compared byte-for-byte across invocations. Focused golden encoding, provenance,
mutation and artifact tests separately protect those contracts.

## Local timing observations

Each selected workflow has two warmups and 21 recorded adapter execution samples
per source process. Independent evaluation runs after each timed call. Output
digests must remain identical throughout. The target-artifact interval retains
target validation, block optimization, provenance, artifact construction and
serialization; ordinary compilation retains its distinct adapter boundary.

| Inspected line-topology workflow | Baseline median (ms) | Candidate median (ms) |
| --- | ---: | ---: |
| Ising 8, ordinary restored | 10.390 | 9.094 |
| Ising 8, target artifact | 206.061 | 171.684 |
| QFT 12, ordinary restored | 39.029 | 33.970 |
| QFT 12, target artifact | 873.755 | 735.558 |

These observations are from one shared macOS CPU host with one numerical thread.
The baseline process ran before the candidate process; the two processes were
not interleaved. Host drift, cache effects and process order are not controlled
well enough to establish significance or stable speedup. The table is a combined
development observation, not isolated attribution to any one optimization or a
comparison against Qiskit/Cirq. Single application samples in the 120-case
correctness corpus are also retained, but are not latency qualification data.
No process memory, cold-start, hardware, held-out or global SDK ranking claim is
made. Formal resource-qualified SDKBench execution remains deferred.

## Validation and reproduction

The combined source passed 803 focused tests with four optional tests skipped,
covering native legality, CZ synthesis, routing, dynamic feedback, IR encoding,
v4 evidence, offline artifacts and the local CPU workflow. The unchanged SDKBench
native adapter suite passed 60 tests. The public API migration snapshot passed.
Earlier local attempts lacked tracked sparse-checkout examples and evidence;
they were incomplete, and are not passing evidence. Restoring those files
enabled the complete focused selection without changing tests or thresholds.
Full default/runtime tiers and complete-tree provenance checks remain CI gates;
the local focused runs do not substitute for them.

Use disposable checkouts at the two source commits above and an unchanged pinned
SDKBench checkout. Copy this change's
[runner](../../benchmarks/compiler_pipeline_development.py) to an external path.
Run once for each source root, changing `fq-source` and the output path. On Linux,
with the existing dependencies:

```bash
timeout --signal=TERM --kill-after=10s 600s env \
  PYTHONPATH=/path/to/fq-source \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python /path/to/compiler_pipeline_development.py \
  --source-root /path/to/fq-source \
  --sdkbench-root /path/to/unchanged/sdkbench \
  --output /tmp/compiler-pipeline.json
```

The runner refuses the wrong imported FlagQuantum root, tracked compiler changes
or a changed/wrong-revision SDKBench checkout. It records source commits, five
touched source digests, its own digest, runtime environment and all timing
samples. These source hashes identify the changed files rather than replacing
the complete pinned Git tree. The retained macOS execution used an external
600-second process-group watchdog with termination cleanup. A complete report
must contain 120 application outcomes and four latency outcomes; a checkpoint
from an interrupted run is not an accepted complete report.

The initial integration CI rejected the unreviewed evidence field name
`compiler_commit`. The reports and runner now use the existing vocabulary
`implementation_commit`; the SDKBench revision remains `source_commit`.
This is a field-name correction, not another measurement. All numerical values,
source identities, output digests and recorded runner digests are unchanged.
The recorded runner digests identify the original runner preserved in commit
`3495f944c92948678d7bd86a31f76b3a445da8e0`; the current runner emits the corrected
field name and consequently has a different digest. The revision vocabulary
gate and its assertions remain unchanged.

The correction passed the revision-field vocabulary test. A local run of all
58 revision tests initially had two failures: a missing tracked example asset
manifest in the sparse checkout, and 16 origin-declaration conflicts caused by
revisions available through this development repository's additional refs.
Restoring the manifest allowed the other 57 tests to pass. The complete-tree
origin-declaration test remains unqualified locally and must pass in CI; its
declarations and assertions were not changed.

The supported user workflow remains the public circuit-to-compiler path shown
in [the routing study](COMPILER_SABRE_DEPTH.md#existing-user-workflow). The
integration is intended to reduce avoidable CPU compiler work and preserve
classical semantics while keeping that workflow unchanged. None of these
measurements establishes broad compiler superiority.
