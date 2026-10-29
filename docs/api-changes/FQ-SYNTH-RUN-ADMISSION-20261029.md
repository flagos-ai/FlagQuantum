# Synthesis Program admission to fq.run

Status: proposed for API-owner review; implementation is not authorized by this
record. The user requested continuation after merging CS06-C and previously
reviewed the fq.run(program) direction. This PR makes that decision concrete;
it does not claim API-owner approval or publish a new namespace. Classification:
single_device_fast_path. Current main remains on fq.run(program.execution).

## User journey and proposed surface

The intended public workflow, **not implemented by this proposal**, is:

```python
import flagquantum as fq
from flagquantum import synth

def polynomial(x: int, y: int) -> int:
    return x * x + y

program = synth.synthesize(
    polynomial,
    inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
    input_state=synth.InputState.basis(x=3, y=2),
    max_qubits=18,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y", "result"))))
# {(3, 2, 11): 1.0}
```

Synthesize returns a complete Program, not only its ExecutionPlan. Program keeps
the circuit/report, checked plan and logical output mapping together. Ordinary
users never access execution or construct ExecutionOptions. execution remains an
expert inspection route, and circuit remains the explicit expert transformation
route. No samples/counts or Program serialization/.fq contract is added.

## Concrete type ownership

The proposed public domain is flagquantum.synth. Its Program is the same class
object as a Services-owned, public implementation type in
flagquantum.services.synthesis.Program, replacing the private _synthesis module.
The public synth package is a small re-export surface; it owns no duplicate
builder, class, validator or executor. Move the existing implementation rather
than wrap/subclass it. Migrate private imports in the same implementation PR;
the unreleased private route gets no permanent compatibility adapter.

The initial namespace exports are ConstructionBudget, InputState, Program,
SynthesisError, UInt, qfunc and synthesize. The proposed classification is a
maintained domain extension; its public surface/compatibility contract must be
approved and recorded together with the root input admission. A non-underscore
package name alone does not make it public under PUBLIC_API_POLICY.md. Until
that registration, the current private workflow remains internal. This proposal
does not invent an untracked experimental namespace outside the governed surface.
Program is factory-created; its current private dataclass constructor, field
layout, allocation records and fingerprints are not proposed public promises.
The required user surface is circuit, report and probabilities; execution and
candidate inspection are optional expert surfaces. No new Core executable type,
artifact envelope, serializer, registry or protocol is justified by this case.

At the root facade, type checking uses a SynthesisProgram alias for the admitted
public Program. The proposed run input union adds SynthesisProgram after
LindbladPlan. All existing keyword parameters, defaults and return type remain
unchanged. This is an additive **accepted-type/behavior change** even if root
exports do not change. The machine proposal records exact before/after signatures.

## Dispatch and bound execution semantics

The root facade alone recognizes the exact approved type and obtains its already
checked execution plan. It then continues through the existing ExecutionPlan
branch, returning the existing ExecutionResult unchanged. No recursive public
run call, replanning, recompilation, resynthesis or custom execution path occurs.
Runtime and Compiler gain no reverse dependency on Services or synth.

Reuse the existing lazy module-presence dispatch pattern used for domain plans.
Constructing a Program necessarily loads its owning module; unrelated Circuit
calls must not eagerly import synthesis, Services or new dependencies. An object
merely exposing execution/to_ir is not admitted. The initial factory-created
profile admits the canonical class itself, not user subclasses overriding plan
resolution. Do not add Program.to_ir to
make generic circuit coercion appear to work: that discards bound execution
semantics and can invalidate basis, resource and layout checks.

For a Program, options, outputs, noise_model, compiler, target, target_qubits,
shots and name must all be None. Even an equivalent non-None override is refused,
consistent with the existing sealed-plan option policy and this initial exact
CPU profile. Reject overrides before resolving/validating the plan or executing.
To request new synthesis constraints/targets, synthesize again. Other input types
retain all existing override behavior. No new fq.plan/compile/submit admission
for Program is implied.

The existing Program validation remains authoritative for source/stage/port and
Compiler-to-Runtime binding. A remains authoritative for sealed executable,
controls, basis, topology, constraints and result association. Failure propagates
without fallback or repair. Probabilities continues to use actual supplied state
and bounded physical named ports. Result.plan stays ExecutionPlan; no result
fields, metadata schema or serialized plan fields change.

## Compatibility and alternatives

- Returning ExecutionPlan from synthesize loses the coherent home for named
  outputs, resource reports and source/candidate bindings; rejected.
- Keeping execution mandatory is a valid current expert path, but leaks an
  avoidable implementation concept into the ordinary journey.
- Generic execution duck typing or a provider registry widens admission beyond
  this verified case; rejected. A second Program wrapper/executor is unnecessary.
- A to_ir coercion silently re-enters planning; rejected.
- Inheriting from ExecutionPlan conflates a synthesis product with one of its
  components and promises unrelated construction/serialization behavior; rejected.

Existing Circuit, CircuitIR, ExecutionPlan, Ising and Lindblad inputs remain
unchanged, including errors and lazy imports. No stable route is removed, so no
deprecation/removal version is proposed. The private CS06-C spelling migrates
with all live callers. Release notes must label the additive input admission,
initial exact CPU scope and namespace maturity without claiming general Qmod,
hardware-native or independent usability evidence.

## Exact implementation and review scope

After approval, the implementation PR must update the Services public module
and synth facade, the root input annotation/lazy dispatch, examples, typing,
ownership classification and release documentation. It must coordinate the
execution-plan and observable-output candidate signatures with the existing
public-api checker, whose proposal-092-specific root-union assertion is currently
exact. Historical baseline and previous approval records stay intact. Do not
regenerate snapshots, weaken the checker, set an approval flag automatically or
treat green implementation tests as approval.

This proposal PR changes none of those production/protection surfaces. Its
reference-consumer tests use today's actual Program.execution and existing
Runtime to demonstrate the proposed underlying semantics. Recorded-plan replay
substitutes behind that test consumer; it is transport conformance rather than
an independent numerical oracle. The reference adapter is test-only and is not
an implemented fq.run(Program) feature or public API stub.

## Acceptance and gates

1. Production fq.run(program) and the existing explicit-plan route have identical
   plan identity, complex amplitudes, resources and named probabilities for both
   schedules, logical and actual nonidentity line routes.
2. Forbid planning, compilation and synthesis while executing the admitted
   product; prove ordinary Circuit import/execution does not load Services/synth.
3. Reject all eight override families and execution-like unrelated objects
   before resolving/executing; propagate stale stage/port/control failures.
4. Preserve gradients, zero workspace, result type/plan and bounded named reading;
   serialized existing ExecutionPlans still replay without a new artifact schema.
5. Replace live plan resolution with recorded replay without changing its
   consumer; keep real numerical references separate.
6. Existing public API/contract, old plan serialization, Circuit, Ising/Lindblad,
   Services, compiler and local runtime regressions pass on the new head.

API-owner naming/type/override and namespace-classification approval is pending.
At least two independent
developers' real task/time/blocker/help records remain absent and are required
before public synthesis milestone claims under revision six. A maintainer
must explicitly decide rollout ordering; agent tests do not fill those records.
Only after these gates may the accepted type and namespace be activated through
the normal protected contract process. This PR may be merged as a proposal
without asserting that approval or implementation is complete.

The concrete rollout is proposal review, explicit API-owner authorization,
implementation on a reviewable feature branch, two independent developer trials
on that branch, and then activation/release review with fresh CI. Approval of a
proposal's scope must be recorded separately from final release readiness.

Trial tasks are the basis polynomial (x=3,y=2), the 16-component uniform joint
distribution, constraint-driven pressure scheduling (nine qubits versus 42
operations), line-profile named reading, and an infeasible request repaired from
actual checked evidence. Each developer records the tested commit, start/end
times, code, help used, blockers and actual outputs, independently of the
implementation agent. Records are currently absent; no timing or usability
score is inferred from automated tests.

## Proposal verification

Fifteen new proposal-only cases plus existing plan/namespace/public API cases
passed (45 total). Expanded A/B/C, live numerical, serialization, fixture
isolation and runnable-example regression passed 238 cases. Production source,
public manifests/checkers and existing frozen contracts are unchanged.
The initial test run exposed a test-only lazy-alias monkeypatch restoration
issue; resolving the root alias before patching its owner fixed isolation, and
the combined regression passed. No production behavior was changed to hide it.

Architecture, current public API baseline, dependency policy, construction
acceptance, vocabulary, formatting, language, scope and links passed. Local
smoke/unit stopped at the same three native macOS QFT fusion-count assertions:
4346 passed, 121 skipped, three failed in 193.86 seconds (`--maxfail=3`). These
observed 44 passes against 28 in test_controlled_phase_graph.py; Simulation and
assertions are unchanged. This local default tier is not green. New-head Linux
CI remains required for this proposal; merged #961 checks are not reused.
