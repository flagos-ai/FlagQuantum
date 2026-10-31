# Public predicate phase delivery

Date: 2026-10-31. Unreleased development evidence. After #974 merged as
1afca2cbc9128f070b6637f75b2818e0f6b2ae61, the human explicitly authorized the
concrete public output_mode workflow. The historical private proposal is intact;
[authorization](../api-changes/FQ-SYNTH-PHASE-IMPLEMENTATION-20261031.md) and the
[active contract](../../contracts/synthesis-phase-v1-candidate.json) record the
exact additive delta. No independent developer trial or GitHub review is invented.

## User outcome

Users write a one-bit unsigned computation and declare input widths, preparation,
phase/value effect and resource ceilings. The same function can produce an XOR
result flag or mark a predicate phase. They do not write gates, physical indices,
bitstring decoding or cleanup. Ordinary Python functions remain the primary
entry; qfunc is optional and no additional DSL or executor is introduced.

Save and run a Python file, or run python -m examples.synthesis_phase_oracle:

```python
import flagquantum as fq
from flagquantum import synth


def marked(x: int, y: int) -> bool:
    return x * x + y < 5


program = synth.synthesize(
    marked,
    inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
    input_state=synth.InputState.uniform(),
    output_mode="phase",
    max_qubits=22,
    memory_limit_bytes=64 * 1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y"))))
print(program.report["resources"])
```

Each pair has probability 1/16; its amplitude is -1/4 if x*x+y<5, otherwise
+1/4. Probability equality does not prove a phase oracle. Complete complex-state
references, workspace-zero checks and interference tests supply that evidence.
The example has no private imports, hand gates or decoding. Internal acceptance
may use those lower-level tools to inspect the circuit independently.

## Reuse and exact scope

Services appends output_mode after existing keyword parameters, validates its
source/target constraints before source work, forwards it to the existing
Compiler selector and adapts report/register semantics. Four overloads distinguish
Python value/phase, explicit native text value, and native file value. Defaults,
function=, immutable inputs, helpers/constants, preparation validation and memory
budgets remain as before. Native reports gain the effective value mode too.

Compiler capture, SSA, one-bit verification, finite candidates, retain/recompute,
exact prefix-adjoint cleanup and resource checking are unchanged. Runtime's
bound plan, fq.run and Program.probabilities remain unchanged. No alternate IR,
wrapper, executor, schema or facade export. The active Program/run candidate
tracks the authorized final keyword; historical Qmod and phase proposals and
baseline snapshots remain byte-for-byte unchanged. The checker validates the
exact successor delta and rejects unauthorized scope/approval changes.

Value mode stays the default with its existing result register and targets.
Phase requires a verified one-bit Python result and basis=None/topology=None.
Native phase and explicit target conversion/routing are rejected before reading,
capture or planning. A multi-bit return fails after capture and before planning;
there is no integer-to-Boolean coercion. Input preparation is outside U, and all
workspace allocated by the factory is zero and returned to zero. Dirty unknown
external inputs are not admitted by this workflow.

Phase registers contain inputs only. The unchanged default reader requests
result and refuses with available names; users select input names explicitly.
The report records output_mode, result_width=0, exact phase scope and common
source/actual resource costs. arithmetic_resources is omitted for phase and
retained for Python value. Workspace includes the predicate temporary. State
buffer limits are planning limits, not process peak memory guarantees.

General line phase remains blocked by existing exactness/layout evidence: the
private polynomial line probe passes, but two-bit equality line refuses lost
routed-SWAP evidence. Existing basis conversion has projective paths. No guard,
phase tolerance or global-phase schema is weakened or changed. Generic controlled
Circuit legalization, within-apply blocks, arbitrary angles, Qmod phase,
GPU/QPU/gradient support and all 216 originals remain outside this admission.

## Acceptance and publication gates

Acceptance runs the public source-retaining Python entry, bound Program and real
CPU execution: both objectives, all 16 input pairs for less-than polynomial and
equality, full complex preparations, constant and direct one-bit returns,
helpers/constants/qfunc, clean workspace, interference and double application.
It also checks actual costs/plan identity, default/explicit value equivalence,
source/target/type early refusals, finite-search/resource/memory refusals,
static overloads, public example CLI and API-contract mutation rejection.

The public example actually executes 18 qubits, including 14 workspace, with
273 operations and depth 178. Its planned complex128 state buffer is 4 MiB.
Source and actual resources remain separate; no process peak or optimality claim.

Final focused public/API verification passed 89 checks, including strict nested
contract-type mutation rejection. The source, private phase and comparison
bridge passed 67 checks, overlapping the broader regression below. Existing
regression passed 1383 checks with 21 optional-dependency skips, including all
six original-size public file/text runs of the three unmodified Qmod originals.
The separate public API candidate suite passed 12 checks. The disjoint focused,
regression and API candidate sets total 1484 passes and 21 skips. A focused
coverage subset passed 37 tests (four larger basis-input cases excluded after
real execution), with 76% aggregate coverage of the shared Services module;
this is neither repository coverage nor a claim that older paths are all covered.

Strict mypy passed 876 package files plus the public example. Minimum Black
24.1.0, current Black, Ruff, API, architecture, dependency, team scope,
capability/generated documentation, vocabulary, language and links passed.
Historical phase/source proposals, the Qmod admission and both API snapshots
were checked byte-for-byte unchanged.

Local default verification passed 4562 tests with 121 skips, stopping at the
same three pre-existing macOS native-QFT gate-count failures (44 versus 28).
Simulation and their expectations were unchanged; local default is not claimed
green. CI status is recorded on the PR after publication.

Independent human trial records remain 0/2 and release_authorized=false. Public
development admission does not certify release readiness or hardware/gradient
performance.
