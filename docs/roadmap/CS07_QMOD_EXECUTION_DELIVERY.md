# CS07-D checked source execution and original-size acceptance

Date: 2026-10-30. Baseline: merged #967,
`13f123dda0aa8a671f558bf18a56d91ecdb8e0af`. Classification:
`single_device_fast_path`. This implements the private Services/integration
portion of the [shared admission contract](../api-changes/FQ-SYNTH-CS07-20261029.md).
No public export, signature, serializer, source format or Runtime executor changes.

## Existing user experience

The normal Python workflow remains the same. Save definitions in a source file:

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
assert dict(program.probabilities(fq.run(program))) == {(11,): 1.0}
```

Run `python -m examples.algorithms.synthesis_trial` for basis, superposition,
constraint recovery and line routing. Users design no gates, physical ports or
bit-string decoder. This PR reuses that workflow's execution and reading
authority for an independent producer; it does not publish a new source API.

Contributors can inspect the private native path without an SDK or cloud:

```python
import flagquantum as fq
from flagquantum.ecosystem._qmod.adapter import SUPPORTED_SOURCE_VERSION, parse_qmod
from flagquantum.services._synthesis_registers import synthesize_registers

source = "qfunc main(output q:qbit) { allocate(q); H(q); }"
imported = parse_qmod(source, source_version=SUPPORTED_SOURCE_VERSION)
program = synthesize_registers(
    imported.program, imported.functions, memory_limit_bytes=512 * 1024**2,
)
result = fq.run(program)
assert set(program.probabilities(result, registers=("q",))) == {(0,), (1,)}
```

These underscore imports are private development seams, not a supported user
entry or an instruction to make users manage Compiler records. Source output
names remain unchanged; native callers select them explicitly. The existing
public default `registers=("result",)` is unchanged, so an absent result name
is rejected rather than invented or silently renamed.

## Ownership and reuse

Ecosystem still owns bounded source parsing, source locations, native roles and
version context. Its owned HybridProgram and expression definitions enter
Compiler's existing verifier, retain/recompute selector, resource constraints,
target product and layout authority. Services' private register workflow consumes
only those owned records. It does not parse text, execute it, import vendor SDKs
or introduce another canonical representation.

Both Python arithmetic and fresh-register construction now use the same private
`synthesis._complete` factory. It admits one selected candidate, plans exactly
once with exact local CPU/torch/complex128 options, checks actual executable
identity/resources/layout and returns the existing Program. No synthesis call
executes. `fq.run(program)` and `Program.probabilities` are unchanged consumers.
The source profile reports live output widths and Compiler semantic identities;
it does not label quantum constants as classical inputs or invent one arithmetic
result. The Python report retains its original fields and meanings.

For this private register profile, Program.arithmetic and arithmetic_resources
refer to the Compiler RegisterCandidate's complete untransformed construction,
including ordered preparation; they are not isolated arithmetic-only costs.
Source_resources and actual resources show the whole-program comparison. A
future public source proposal must settle this presentation explicitly rather
than infer a common arithmetic-only meaning from the reused field name.

The native seam requires a valid explicit memory ceiling. Runtime validates the
option and seals it into its existing decision. Services checks the validated
plan's state_bytes before returning an admissible Program. The existing planner
can describe an over-budget state plan, so merely passing an option is not
evidence of admission. This check bounds the planned state buffer, not total
process RSS, kernel temporaries, allocator reservations or proof of host capacity.
Construction ceilings and readout ceilings remain distinct controls.

Original tests preflight at most 24 qubits and four reference-sized state buffers
within 1 GiB before execution, with a separate 2 GiB Runtime memory option.
That is a test-harness allocation bound, not a measured peak-memory claim. The
existing 22-qubit routine conformance helper is unchanged. Larger original tests
are explicitly marked slow and integration; they are not reduced source probes.

## Readout fix and refusal behavior

The existing named reader checked max_bits/max_keys/max_amplitudes but omitted
the max_marginal_wires metadata when invoking the existing measurement engine.
Consequently a legal 9/12-bit named request still hit that engine's default
8-bit ceiling. Pass the already-validated max_bits through that existing option;
do not change Runtime defaults or create another marginal implementation.
A nine-bit scalar regression proves both successful reading and max_bits=8
refusal. The original public defaults and explicit read-budget checks remain.

Both producers use the same checked plan and physical named ports. Tests reject
wrong result identity, altered names, unnormalized state, insufficient read or
memory budgets, infeasible final constraints and incomplete candidate search.
Synthesis never runs and rejected searches never plan. Source diagnostics and
source-version context remain in the external adapter; Compiler semantic
identities are not authentication of native bytes. Original manifest/license
checks remain independent and unchanged.

## Original-file evidence

All three preserved files use upstream revision
`2bfd835e73b6882fcc515a725098b04766c238e7`. Source and MIT-license bytes remain
unchanged. Source integrity, parsed, lowered, executed and complete-state verified
counts are each **3/3**. This supersedes CS07-C's historical 1/3 execution count.

| Original | Output widths | Total / workspace qubits | Source / actual operations | Actual depth | State buffer |
| --- | --- | --- | --- | --- | --- |
| X.qmod | q:1 | 1 / 0 | 1 / 1 | 1 | 32 B |
| synthesis_tutorial.qmod | x:3, y:6 | 23 / 14 | 379 / 379 | 288 | 128 MiB |
| multiplication_2vars_example.qmod | a:3, b:3, res:6 | 24 / 12 | 279 / 275 | 214 | 256 MiB |

The existing selector chooses retain for these default requests. These counts
refer to the exact CPU executable, including preparation, arithmetic and cleanup;
they are not hardware-native costs or optimality claims. All logical/physical
output maps are identity for these originals. Routed synthetic tests separately
exercise real SWAP layouts and named reading with both scheduling objectives.

The independent reference for X is amplitude one at q=1; tutorial has all eight
amplitudes 1/sqrt(8) at (x,x*x+1); product has amplitude one at (4,5,20).
Every complex amplitude is compared, with no removal of global phase, using
residual norm <=1e-12. Mapped workspace nonzero probability is separately
<=1e-12. All live outputs stay present. The exact named marginal is checked
against squared analytic amplitudes and the larger reads explicitly authorize
2**24 amplitudes. Neither probabilities alone nor a hand-written replacement
circuit counts as original verification.

Run `python -m pytest tests/integration/test_synthesis_qmod_originals.py -q -s`.
Each passing case prints ORIGINAL_ACCEPTANCE with source identities, live
logical/physical ports, resource reports, buffer size and separate layer counts.
The 19 routine workflow cases plus three originals passed together (22 cases,
73.37 seconds). Package-wide strict mypy passed 874 source files.

Expanded regression passed 1,118 cases with three skips (68.20 seconds), including
the maintained Python user workflow, public admission, construction acceptance,
Services checks and open-source golden paths. Default smoke/unit reached 4,480
passes and 121 skips before the same three existing local macOS QFT fusion-count
failures stopped it (183.55 seconds, --maxfail=3). They observe 44 instead of 28;
no Simulation code or numerical expectations were changed. The full local
default suite is not green. Black 24.1.0 checked all 2,234 Python files unchanged;
current Black, Ruff, public API, architecture, dependencies, capability metadata,
generated documentation, repository language, vocabulary and changed-doc links
also passed. Independent human trials and release approval remain pending.

## Remaining scope

The [public source admission proposal](../api-changes/FQ-SYNTH-QMOD-ADMISSION-20261031.md)
now makes input dispatch, resource scopes, memory defaults, errors and the
future public acceptance concrete. It remains a proposal, not an activated API.

Native source entry remains private. Before exporting one, prepare a concrete API
proposal covering names, source context, constraints, file/text ownership,
diagnostics and the same Program/run/read journey. No run_qmod, public qmod
namespace, .fq loader or second Program is introduced. Existing unsupported
arrays/views, Input/Const, free/drop, helper functions, general standard library
and other native syntax remain rejected as recorded by CS07-C.

Three pinned originals do not imply all 216 programs, complete Classiq language
or SDK/cloud/hardware compatibility. Two independent developer trials remain
0/2 and release approval remains false; automated execution is not human trial
evidence. A public-entry review and the remaining dependency-ordered library
roadmap are separate next steps.
