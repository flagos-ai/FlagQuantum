# Predicate phase construction and proposed public admission

Date: 2026-10-31. The human requested merging #972 and proceeding to the next
step. #972 merged as d6df7bca0d5f68b760518af023f00f0669b66145 after its exact
head bea9c82985bc68ec0eb50b570ed0e3f876d8686c passed 57 checks with three
configured skips. This authorizes the next private Compiler phase construction
and acceptance slice. The new public keyword below is a concrete proposal for
review, not an implemented or approved public API. No approval/trial is invented.

## Private effect and reuse contract

For a verified unsigned function f returning exactly quantum.uint(1), emit
O_f |x,0_work> = (-1)**f(x) |x,0_work>. There is no output flag. Input encodings
are preserved while their relative phases intentionally change. Workspace is
fresh/caller-asserted zero; unknown or dirty workspace is not certified.

Reuse existing SSA, binding checks, dependencies and both schedules. Capture the
actual selected forward gate prefix U at the point its predicate is available,
apply exact Z to that bit and reverse that same prefix as its exact adjoint.
The closed prefix consists of self-adjoint X/CX/CCX; another gate kind must be
refused until an exact adjoint rule is admitted. Do not re-run selection or
resynthesize another implementation for cleanup. Recompute inverses mirror the
actual physical allocation history, including earlier compute/cleanup events.
All gates/events count against the existing construction budgets.

Private output_mode="phase" requires an empty output binding and a one-bit
return. Mapping includes only logical inputs; predicate temporaries count as
workspace. Input preparation occurs before the oracle and is never inverted as
part of U. It may touch input qubits only. Target lowering, resource checks and
selection are the existing pipeline; source/actual costs cover complete gates.
No new IR, executor, public wrapper or public export. Qmod grammar is unchanged.

## Proposed public contract, pending review

Append output_mode: Literal["value", "phase"] = "value" to synth.synthesize,
after the existing keywords, retaining positional/function= calls and every
existing default. Python Definition accepts both modes; native string/PathLike
inputs accept "value" only and refuse "phase" before file acquisition. Invalid
mode types/values fail before capture/planning. No public type or facade export.

"value" preserves the present unsigned XOR result. "phase" requires a verified
one-bit result, including existing comparison predicates and optional bool return
annotations. Existing immutable inputs, constants/helpers and known preparation
remain supported on the native logical CPU target. Phase mode initially requires
basis=None and topology=None; reject other requests before capture/planning.
Use four overloads: Python value, Python phase (basis/topology None-only),
explicit Qmod text value, and PathLike value. Python inputs remain required.
There is no classical branch, measurement, arbitrary angle,
controlled-block DSL or implicit integer truthiness.

Both modes use the same complete Program factory and bound plan; execution
still uses fq.run(program). Proposed report addition output_mode records the
effective mode, with the existing fields retained. Value scope stays
actual_cpu_preparation_arithmetic_cleanup; phase scope is
actual_cpu_preparation_predicate_phase_cleanup. Workspace costs include all
predicate temporaries. Preserve source_resources versus actual resources and
the current effective memory limit/origin behavior. Phase reports omit the
pure value evaluator diagnostic arithmetic_resources; Program.arithmetic remains
a private profile-specific diagnostic, not a new wrapper or public promise.

Phase register_widths/registers include input names only. Named probability
reading retains its exact signature and default ("result",); that default must
fail with available names on phase products, so use explicit input names.
Probabilities alone do not reveal a phase oracle. Do not rename an input, add a
fake flag or claim that a uniform probability table proves correct phases.
Program.circuit remains the existing expert composition route; a new high-level
Grover/algorithm consumer is not admitted by this proposal.

## Proposed user code: requires later public implementation

Save as a Python file once the proposal is approved and implemented:

```python
import flagquantum as fq
from flagquantum import synth


def marked(x: int, y: int) -> bool:
    return x * x + y < 5


program = synth.synthesize(
    marked,
    inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
    input_state=synth.InputState.uniform(),
    output_mode="phase",  # Proposed keyword; not available in this PR.
    max_qubits=22,
)
result = fq.run(program)
print(program.report["resources"])
print(dict(program.probabilities(result, registers=("x", "y"))))
```

The oracle changes each input amplitude by (-1)**marked(x,y), returns its work
to zero, and leaves the displayed input probabilities uniformly 1/16. Full
complex-state and interference acceptance below verify the phase; the printed
distribution does not. The existing value-mode predicate example remains usable
as python -m examples.synthesis_predicate. The actual private contributor
workflow is python -m examples.synthesis_phase_oracle, explicitly not a stable
user entry.

## Acceptance

- Both schedules: all 16 basis inputs, complete complex amplitudes on entangled
  inputs and a control-zero reference branch; no quotient by global phase.
- Constant true/false predicates, direct one-bit borrows, helper dependencies,
  unequal input widths and exact inverse-prefix identity.
- Repeated oracle application is identity on clean workspace. Interference
  converts a small oracle's sign into a different observed probability outcome.
- Native logical targets and separate private line success/refusal evidence;
  exact controlled-conjugation phase after target compilation,
  source/actual costs, all-zero workspace and final constraints/budget refusals.
- Existing public signatures/exports, API baselines, value-mode and Qmod flows
  remain unchanged. Public phase calls are refused until separately authorized.

See contracts/synthesis-phase-admission-proposal.json for the exact proposed
signature and pending status. Independent user trials remain 0/2 and release
authorization remains false. General within-apply/control/invert and all-corpus
compatibility require later separate admission.

## Target admission finding

The existing basis compiler documents projective paths (notably cphase) and the
IR global-phase field remains unimplemented. A phase line profile keeps u3, rz,
phase, z and cz native, with optimization disabled; preparation must use plain
native instructions. Controlled conjugation applies control to the marker only,
leaving the selected compute/inverse unchanged. It avoids projective cphase
conversion caused by controlling every compute/cleanup gate separately.

Private polynomial phase line execution passes full complex checks. Two-bit
equality line construction is still explicitly refused by the existing routed
SWAP-witness guard after normal Runtime optimization. The guard is retained and
its refusal tested, not counted as a successful phase execution. Therefore the
public proposal admits native logical CPU phase only. General line phase and
projective target/control pipelines require separate routing/phase evidence
work before admission; no claim of generic controlled Circuit legalization.
