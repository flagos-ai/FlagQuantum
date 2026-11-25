# Original PHASE through the existing execution path

Follow #1001 at f79a806ba1b456b5d3d79eb636fbad2b107e0bd7 after 57 successful
checks and three configured skips. See the
[contract](../api-changes/FQ-SYNTH-CONSTANT-PHASE-GATE-20261102.md).

## User workflow and exact phase convention

```python
import flagquantum as fq
from flagquantum import synth

program = synth.synthesize(
    "qfunc main(output q:qbit) {allocate(q);H(q);PHASE(1,q);}",
    source_format="qmod",
    max_qubits=1,
    max_workspace=0,
    memory_limit_bytes=1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("q",))))
print(result.statevector())
```

Run `python -m examples.synthesis.tutorials.qmod.qmod_phase_gate.main`. Probabilities remain 1/2,
but amplitudes are 1/sqrt(2) and exp(i)/sqrt(2), approximately 0.7071 and
0.3821+0.5950i. PHASE(theta)=diag(1,exp(i*theta)), as defined by the retained
Classiq standard-gates reference. RZ differs by a global phase and cannot be
substituted without preserving it. Basis-one and reference-entangled probes
compare absolute complex amplitudes; original basis-zero execution alone would
not establish the convention.

On CPU complex128 / Torch 2.13.0 the complete prepared example compiles to
one U3 instruction, one allocated/used qubit, depth one, no workspace and a
32-byte planned state buffer. Existing optimization combines H and phase;
source operations are not presented as the final physical gate count.
The unchanged original PHASE.qmod itself has one phase instruction with theta
1.0, one qubit, depth one, zero workspace and 32 planned state bytes. No state
capacity, runtime performance, optimality or hardware claim follows.

## Ownership and bounded source profile

Compiler changes two existing register files. qreg.gate with gate=phase requires
exactly gate/index/angle; angle is a finite Python float in radians. Other gates
retain exactly gate/index. It emits existing Core Instruction phase with
params={"theta":angle}; no new opcode, IR type, gate kernel, executor or public
API. Initialized storage, numeric bit indices, Const refusal, effect ordering,
mutation invalidation, helper expansion, budgets and source coordinates remain.

Ecosystem changes only adapter.py. Uppercase PHASE(k,q) accepts an integer
literal in 0..2**53 and initialized scalar qbit, converts its exact value to a
float and delegates validation/lowering to Compiler. Decimal, negative,
symbolic, computed and quantum angles remain refused; numeric/array targets
are refused. PHASE is now a reserved intrinsic, so a helper cannot shadow it.
This is a bounded addition to the unreleased profile, not general CReal support.
Lowercase phase(expression,pi), within/apply actions and Const gate permissions
remain distinct and unchanged. No source evaluation, SDK, new dependency,
parallel role checker, target allocator or backend selection is introduced.

## Original-source evidence and coverage

PHASE.qmod retains all upstream bytes at commit
2bfd835e73b6882fcc515a725098b04766c238e7. Git blob
8163d1ad91e916fcc5fa6c5c5e71f61d934efbed and SHA-256
8cd253f41e1ae55f32592e1a8af20c09409d970e365d4e84b5e0eed06bcca035 are pinned
in the additive phase_gate_sources.json and independent test assertions. The
old manifests, fixtures and shared MIT license remain unchanged.

Both original file/text inputs execute at the original one-bit size and compare
the entire state and actual resources. Together with coherent semantic probes,
this raises local verified original paths from six to seven of 216 (3.24%),
and distinct source blobs from five to six of 208 (2.88%). It does not imply
seven algorithm families or complete standard-library support. A fresh
whole-corpus frontend-only scan admits seven and refuses 209; only the PHASE
path changes admission. First qmod.call blockers decrease from seven to six.
Frontend admission remains separate from execution and original-source counts.

## Verification boundaries

The final focused suite has 53 passing cases in 0.69 seconds: finite angle and
exact attribute validation, indices/initialization, original bytes/license and
file/text execution, both objectives, coherent reference states and absolute
basis phases, native/owned-record replacement through the same checked Program,
helper composition and lowercase phase separation, unsupported angle/role/type/
scope/shadowing refusals and actual operation/width/workspace/memory guards.

Initial lowering incorrectly supplied positional parameters to Instruction;
CoreIR requires its existing named theta mapping, which is now reused. The
first producer-replacement fixture compared raw gates to an optimized Program;
it now uses the same checked Program factory, with its required explicit memory
budget. These failed runs are not counted as passing or used to justify changing
CoreIR, exceptions or stable APIs. The initial named-mapping correction missed
a multiline formatted expression and was fixed before the successful run.

Strict mypy 2.4 passes 879 production/example sources. Black 26.10.1, Ruff,
architecture, unchanged API snapshot, dependency policy, generated docs,
maturity, language, local links, whitespace and the original seven source/license
digest checks pass.

Required default smoke/unit: 5051 passed, 121 skipped and the same three existing
local Simulation QFT fusion assertion failures in 212.38 seconds, stopped at its
configured three-failure limit. These still expect 28 state applies and observe
44. No Simulation implementation or assertion is changed; the default tier is
not green. Log: /private/tmp/fq-phase-gate-default.log.

Broader Compiler/Ecosystem/synthesis/API regression passes all 2601 cases in
607.27 seconds. It includes the initial 49 new cases; four subsequently added
shadowing/target-basis cases are covered by the final focused 53-case run. The
production implementation is unchanged between these runs. Two source-overload
mypy-cache fixtures are excluded; whole-package strict typing passes separately.
Log: /private/tmp/fq-phase-gate-regression.log.

The extra target checks preserve basis-one global phase and entangled relative
phase with a named basis including phase and line routing. A basis excluding
phase, with no verified existing decomposition, fails closed rather than losing
a global phase or choosing an unrecorded approximation. Initial probes requested
unsupported bases; their refusals were retained as an explicit negative test,
and positive probes now name the gate they require. No target decomposer changes.
Independent trials remain 0/2 and release_authorized=false. No arbitrary Python,
general Qmod, consuming roles, arrays, signed/fixed-point data, GPU/QPU admission
or release promotion is inferred from this gate.

## Subsequent native-first extension

The [RZ/native-phase follow-up](SYNTH_LITERAL_RZ_DELIVERY.md) adds fixed-decimal
angle literals to PHASE/RZ and preserves the existing integer and predicate
profiles. UP-00 withdraws its unreleased float-return prototype and keeps the shared
angle/RZ import slice. The earlier integer-only PHASE source restriction above
records this delivery's historical boundary rather than the extended source
profile. See the [unified contract](../api-changes/FQ-SYNTH-UNIFIED-PROGRAM-UP00-20261102.md).
