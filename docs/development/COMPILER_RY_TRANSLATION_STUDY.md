# Shorter RY translation development study

Status: rejected under the predeclared complete-workflow admission guardrails.
No production source, dependency, public API, target contract, replay validator
or SDKBench policy changes are retained.

## Candidate and mature implementation inspection

Installed Qiskit 2.6.2's standard equivalence library exposes the exact word
`Sdg RX(theta) S` for RY. FlagQuantum's existing word is
`Sdg H RZ(theta) H S`; its existing RX rewrite is `H RZ(theta) H`.
The candidate replaces just that RY rule in the existing immutable rule mapping,
using native Instruction objects and the original parameter object and metadata.
This is reuse of a mathematical equivalence pattern, not copied upstream code
or a port of Qiskit's accelerated equivalence library. Existing target predicates,
recursive search, ranking, Euler alternatives, resource limits, independent
replay and evidence construction remain authoritative. No upstream runtime
object enters the IR and no new production dependency is proposed.

The potential advantage is fewer intermediate leaves when the target can execute
RX; targets without RX still use the existing recursive translation. This
plausible reduction alone does not establish complete-workflow benefit or
universal semantics for symbolic, trainable, batched or annotated inputs.

## Fixed control and execution correction

The driver fixes source `96e972b9cc1dff728455dcc9ba84b48f730d328d` and unchanged
SDKBench `5ba03de33f8a9193f86b26d2cb3fa841ede7c98a`. The source is the unmerged
capacity candidate in PR #1327, not integrated main or a release. Six inspected
external inputs and both existing native workflows yield twelve paired cases.
Two warmups and nine retained alternating samples per arm give 22 adapter
calls per case. Each result passes the existing independent evaluator. These
known development fixtures do not supply held-out generalization.

Before execution, the driver requires no median ratio above 1.05 and at least
5% reduction in both DNN artifact cases. The original supervisor incorrectly
applied one 20-second wall budget to the whole 22-call batch. Both DNN artifact
batches were terminated and their failed attempts are retained. Their incomplete
workers returned no intermediate observations; those observations are missing,
not reconstructed. Ten other completed cases are never rerun.

The separately retained recovery changes only those two batch budgets to sixty
seconds. Single-adapter deadlines, input/configuration identities, sample counts,
variant ordering, acceptance and adoption thresholds remain unchanged. It does
not replace either original timeout attempt. This is execution-prerequisite
recovery, not selection among successful timing samples.

## Complete observed results

Across the ten original complete cases and two recovered cases, all 264 calls
pass unchanged independent acceptance. This is 48 warmups and 216 retained timing
samples. Six-wire cases check complete operators up to phase; DNN checks only
three probes. Times below are unprofiled adapter medians in milliseconds from
nine samples per arm; every sample and independent metric remains in the archive.

| Instance | Workflow | Baseline ms | Candidate ms | Ratio | Baseline/candidate CZ | Baseline/candidate two-body depth |
| --- | --- | ---: | ---: | ---: | --- | --- |
| EXT-02-GRID-08 | `flagquantum-ordinary-restored` | 120.622 | 107.727 | 0.893096 | 264/264 | 111/111 |
| EXT-02-GRID-08 | `flagquantum-target-artifact` | 1271.663 | 1240.250 | 0.975297 | 183/181 | 86/84 |
| EXT-02-LINE-08 | `flagquantum-ordinary-restored` | 119.163 | 109.631 | 0.920015 | 270/270 | 114/114 |
| EXT-02-LINE-08 | `flagquantum-target-artifact` | 1256.382 | 1202.597 | 0.957191 | 190/190 | 82/82 |
| EXT-03-GRID-06 | `flagquantum-ordinary-restored` | 35.697 | 35.415 | 0.992093 | 84/84 | 51/51 |
| EXT-03-GRID-06 | `flagquantum-target-artifact` | 333.905 | 299.892 | 0.898135 | 66/66 | 40/40 |
| EXT-03-LINE-06 | `flagquantum-ordinary-restored` | 31.166 | 31.462 | 1.009499 | 108/108 | 72/72 |
| EXT-03-LINE-06 | `flagquantum-target-artifact` | 350.672 | 332.711 | 0.948783 | 90/90 | 58/58 |
| EXT-04-GRID-06 | `flagquantum-ordinary-restored` | 6.642 | 6.677 | 1.005401 | 44/44 | 43/43 |
| EXT-04-GRID-06 | `flagquantum-target-artifact` | 73.968 | 71.292 | 0.963817 | 36/36 | 33/33 |
| EXT-04-LINE-06 | `flagquantum-ordinary-restored` | 6.149 | 6.188 | 1.006336 | 32/32 | 30/30 |
| EXT-04-LINE-06 | `flagquantum-target-artifact` | 68.568 | 66.023 | 0.962886 | 34/34 | 31/31 |

Both DNN artifact ratios exceed the fixed 0.95 requirement, so the candidate
fails admission even though the ordinary DNN cases improve and the grid artifact
emits two fewer CZs. Small regressions and ties remain in the table. Do not change
the threshold, select the faster subset or rerun for a favorable outcome. A
shorter local word need not preserve the numerical trajectory of later block
synthesis; all changed outputs still require the original independent checks.

The shared macOS host used one numerical thread, with no concurrent local test
suite. Process RSS is observed for the combined baseline/candidate batch; it
cannot be attributed to either arm or interpreted as a hard memory cap. These
short known-input observations establish neither significance, cold-start
performance, SDK ranking, production resource qualification nor broad semantic
coverage. Reject production adoption under this study and retain the simpler
current implementation. This does not disprove the identity or prevent a later,
separately justified design with its own prospective controls.

## Existing user workflow and archive

```python
import flagquantum as fq
from flagquantum.compiler.basis_conversion import convert_basis

source = fq.Circuit(1).ry(0, 0.37)
converted = convert_basis(source, gates=("rx", "rz", "cz"))
assert all(gate.name in {"rx", "rz", "cz"} for gate in converted.program.instructions)
```

The current supported conversion remains unchanged. This report adds no
newly supported basis or capability claim.

The [complete archive](evidence/compiler-ry-translation-study.json.gz) contains
both attempts, candidate builder, frozen guardrails, every retained sample,
all independent metrics, audits, errors, timeouts, source identities and drivers.
The driver's temporary paths describe the observed setup. Use Python's standard
`gzip` module to inspect its JSON. No raw credential or host process inventory
is included. Full default/runtime and exact-head CI remain required for the
report branch; semantic acceptance of this rejected candidate is not validation
of an integrated production implementation.

Archive SHA-256: `bec03eaae870cbc97f8cfde768ac6cc9ae34d36677972c84c5263f7f3a8ec9a9`; size 12,875 bytes.

## Report verification

Fourteen documentation-link and public-evidence privacy tests passed (1.14 s).
Archive completeness, all 264 accepted records, two retained original timeouts,
source metadata, SHA-256, English report text and credential-pattern scan passed.
These focused checks do not substitute for complete exact-head CI.
