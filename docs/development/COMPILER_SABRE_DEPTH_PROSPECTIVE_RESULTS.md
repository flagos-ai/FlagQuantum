# Prospective inputs reject the unchanged depth-only SABRE candidate

Decision: reject general admission under the rule fixed before the draw. Keep
the production selector unchanged. One fresh source/target input increases
delivered CZ count, despite an improved ordinary-workflow depth. The later
identity-prefix collector fixes the previously inspected QFT witness but does
not make a depth-only routing objective safe for target-native cost.

## Frozen source and complete accounting

The [protocol](COMPILER_SABRE_DEPTH_PROSPECTIVE.md) and runner were committed at
`61d957a6ec9310f2aa105d94190bdc5e6d5e8ea5` before the single nonce draw. The
[freeze manifest](evidence/sabre-depth-prospective-freeze.json) retains that
commit, runner/protocol/selector hashes and nonce. The candidate selector AST
is exactly the earlier inspected implementation; no policy was changed after
the draw. Production source matches parent PR #1272 head
`553054f4206ff80c9f4d3005556dd2b3329dd567`. Formal SDKBench qualification
remains user-deferred.

The [complete report](evidence/sabre-depth-prospective-applications.json) contains
all 120 baseline/candidate pairs and 240 independent semantic acceptances, with
zero exceptions or unsupported attempts. The maximum semantic residual is
`4.956510152190599e-14`, below the unchanged `1e-8` threshold. Both variants
restore identity output placement. Width six uses complete operators up to
global phase; widths eight and twelve use three probes. These checks do not
prove universal equivalence for the larger cases.

The [input fingerprints](evidence/sabre-depth-prospective-inputs.json) distinguish
51 changed source/target inputs from nine repeated QFT controls, before the two
workflow variants. QFT operations and line/ring/grid edges do not change with
the nonce. Probe-only changes are not new programs. Accordingly, this run has
102 new source/target workflow pairs and 18 repeated controls, not 120 unseen
program pairs. After inspection, the new inputs are development cases and must
not be reused as unseen qualification.

| Input group | Improved CZ/depth pairs | Unchanged | Regressed |
| --- | ---: | ---: | ---: |
| New source/target pairs | 7 | 94 | 1 |
| Repeated source/target controls | 5 | 13 | 0 |
| Complete study | 12 | 107 | 1 |

The [decision record](evidence/sabre-depth-prospective-decision.json) lists every
changed pair. Compare `(full_cz_count, two_body_dependency_depth)`
lexicographically within each workflow/input. Any regression rejects admission;
aggregate improvements do not override that rule. The diagnostic process exited
zero because every semantic execution passed. That exit is not quality
admission: the separate frozen quality rule rejects this result.

## New counterexample

| Eight-wire grid QAOA | Baseline CZ/depth | Candidate CZ/depth |
| --- | ---: | ---: |
| Ordinary restored workflow | 56 / 31 | 56 / 27 |
| Target artifact workflow | 52 / 27 | 56 / 27 |

The target result adds four CZs without reducing delivered dependency depth.
This is a new source/target input, not the prior QFT witness. The ordinary route
looks better, demonstrating why ordinary-workflow depth alone cannot certify
the target-artifact workflow. The evidence establishes a downstream quality
regression; it does not identify the exact native block or synthesis decision
responsible. No fixture exception, redraw or selector retuning was performed.

## Consequences and limits

Do not enable this candidate. Any replacement needs its own fixed study and
fresh inputs. A future bounded target-cost-aware design must preserve an
incumbent and inspect the delivered objective; the current evidence does not
validate such a design. Search work, compilation latency and process-memory
qualification remain separate requirements. No resource, ranking, hardware,
scalability or general superiority claim follows from these results.

The existing public workflow is unchanged:

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

program = compile(fq.Circuit(4).h(0).cx(0, 3), coupling_map=CouplingMap.line(4))
assert program.metadata["routing"]["mapping_restored"] is True
```

Expected: legal routing with restored logical output placement; no new policy
option or default. The study ran locally with one numerical thread and a
600-second outer process-group budget. The unchanged SDKBench generator,
adapters, evaluator, caps and source revision are recorded in the frozen
protocol and reports. No remote job or host configuration change was required.
