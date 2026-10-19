# QBoson QDiffusion acceptance result (2026-10-08)

## Outcome

The frozen QDiffusion acceptance run **did not pass**. The FlagQuantum Kaiwu
adapter, real-provider route, two-host A800 execution, three-seed training,
local ESM2 evaluation, and replay-host portability path all executed, but the
aggregated guided output exceeded the preregistered repeat-ratio limit.

This is a quality-gate failure, not a provider-connectivity or execution
failure. It must not be reported as QDiffusion application acceptance or as a
quantum-speedup result.

## Frozen lane

- FlagQuantum execution revision: `754623d7248eda555cd517677ad6ece575dfe6d8`
- Kaiwu PyTorch Plugin revision: `f047bce7b1077449967bbe9e9fab5741542b48d4`
- Kaiwu SDK: `1.3.1`, account-default assignment, `project_no=null`
- Experiment-config SHA-256:
  `68538085538ce36181fef26b6012a6c41111d8dbf0e17da6bc7b1041d977b825`
- Environment-lock SHA-256:
  `a16cf3280f29d339adcc92d5fc65d9695a99ff65665f75a471ee2402bf52abf0`
- Primary host: `jp-a800-171`, NVIDIA A800, `cuda:0`
- Replay host: `jp-a800-172`, NVIDIA A800, `cuda:0`
- Training seeds: `1701`, `1702`, `1703`

The execution revision remains the preregistered source identity even though
later review commits fixed evidence-consumer compatibility discovered while
processing the immutable records. The original records and their hashes were
not rewritten.

## Provider and quota result

The formal run submitted 442 distinct provider tasks:

- 1 optimization task;
- 441 sampling tasks;
- 4,410 sampling credits consumed in total;
- 1 optimization credit consumed in total.

The final replay used exactly nine SPQC-1000 sampling tasks and 90 sampling
credits. The authenticated Resource Bill changed from 5,670 available / 4,330
used to 5,580 available / 4,420 used. Bill reconciliation supplied provider
task IDs and the SPQC-1000 target omitted by the Kaiwu 1.3.1 runtime records.
Task IDs, credentials, account identifiers, SDK checkpoints, model weights,
datasets, and raw private evidence are intentionally excluded from Git.

## Evaluation result

ESM2 evaluation ran locally on `jp-a800-171` without provider calls. All three
seeds produced valid sequences, and the aggregate guided mean cosine distance
improved slightly from `0.1699131529` to `0.1674295863` (lower is better).

| Seed | Baseline mean cosine distance | Guided mean cosine distance | Baseline repeat ratio | Guided repeat ratio |
| --- | ---: | ---: | ---: | ---: |
| 1701 | 0.1800070008 | 0.1822192868 | 0.0000000000 | 0.0000000000 |
| 1702 | 0.1446917256 | 0.1445365945 | 0.0000000000 | 0.0000000000 |
| 1703 | 0.1850407322 | 0.1755328774 | 0.0000000000 | 0.3333333333 |
| Arithmetic mean | 0.1699131529 | 0.1674295863 | 0.0000000000 | 0.1111111111 |

The frozen threshold allows an absolute repeat-ratio increase of at most
`0.05`. The observed increase was `0.1111111111`, driven by seed 1703, so the
final validator correctly rejected the candidate. Uniqueness remained `1.0`,
length-match ratio remained `1.0`, and invalid-sequence count remained `0` for
both baseline and guided outputs.

## What this run establishes

The evidence establishes that the reviewed FlagQuantum adapter can:

- submit bounded optimization and sampling tasks through Kaiwu 1.3.1;
- reconcile account-bill task identities without changing component hashes;
- train the frozen QDiffusion fixture on `jp-a800-171` using real SPQC-1000
  sampling;
- evaluate retained outputs on A800 `cuda:0` without additional provider
  quota; and
- replay the frozen trained checkpoint from `jp-a800-172` through nine real
  SPQC-1000 sampling calls with no fallback or retrieval resubmission.

It does **not** establish application acceptance, production readiness,
distributed execution, domestic-accelerator support, performance improvement,
or quantum advantage. End-to-end elapsed time was dominated by isolated input
transfer, environment installation, provider round trips, and evidence
processing; this run was not designed as a speed benchmark.

## Next acceptance attempt

A new attempt should first change the model or guided-generation policy to
reduce long repeats, then freeze a new configuration and obtain a new explicit
quota approval. The 2026-10-08 preregistered thresholds must not be relaxed
after observing this result, and the completed records must not be reused as
if they belonged to the new experiment.
