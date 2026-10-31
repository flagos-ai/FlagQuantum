# Qmod source admission to constrained synthesis

Status: proposed for API-owner review. Implementation and release are not
authorized by this record. Baseline: merged #968,
`2d0ab6fd12c8e54b76abbd16a37067b265642bce`. Classification:
`single_device_fast_path`. The user requested this proposal after reviewing the
two-PR sequence: a concrete contract first, implementation after review.

This PR changes no production entry, current candidate contract, public API
checker or snapshot. Its machine-readable review contract is
[synthesis-qmod-admission-proposal.json](../../contracts/synthesis-qmod-admission-proposal.json).
Passing tests and merging documentation are not fabricated API-owner review,
independent developer trials or release approval.

## Goal and user journey

Users express computation and quantum preparation, declare domains and resource
ceilings, and receive one checked Program. They design no gates for arithmetic,
choose no physical ports or workspace schedule, and decode no bit strings.
Source-retaining Python functions remain the primary high-level workflow; native
Qmod is an additional compatibility input to the same construction authority.

The following Python path already works today, from a source file:

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

The proposed file path below is **not implemented by this PR**:

```python
from pathlib import Path
import flagquantum as fq
from flagquantum import synth

program = synth.synthesize(Path("synthesis_tutorial.qmod"), max_qubits=24)
result = fq.run(program)
distribution = program.probabilities(
    result, registers=("x", "y"), max_amplitudes=16_777_216,
)
# {(0, 1): 0.125, (1, 2): 0.125, (2, 5): 0.125, (3, 10): 0.125,
#  (4, 17): 0.125, (5, 26): 0.125, (6, 37): 0.125, (7, 50): 0.125}
```

This uses the unchanged original three-bit tutorial. The larger read ceiling is
explicit because its actual 23-qubit state exceeds the existing 2**22-amplitude
reader default. Users authorize the stated read cost; they do not reconstruct
the physical layout. Budget errors must show the required cost and applicable
limit. Do not silently enlarge a read budget based on input format.

The proposed text path is also **not implemented**:

```python
program = synth.synthesize(
    "qfunc main(output q:qbit) { allocate(q); H(q); }",
    source_format="qmod",
    source_name="bell_component.qmod",
)
result = fq.run(program)
distribution = program.probabilities(result, registers=("q",))
# {(0,): approximately 0.5, (1,): approximately 0.5}
```

No additional source wrapper, public loader, parser result, runtime protocol,
qmod namespace or Program subclass enters the user journey. Optional qfunc on
Python functions stays optional. Source qfunc main is native grammar, not a
requirement to adopt a decorator in ordinary FlagQuantum Python code.

## Proposed signature and input resolution

Preserve the positional/keyword name function, because existing callers may use
function=polynomial. Broaden its type to Definition | str | os.PathLike[str].
Change inputs to Mapping[str, UInt] | None = None to admit sources; typing must
retain a Python overload requiring inputs. All other current parameters retain
their order, defaults and meaning. Append only these keyword-only parameters:

```python
source_format: Literal["qmod"] | None = None
source_name: str | None = None
source_version: str | None = None
memory_limit_bytes: int | None = None
```

The JSON records complete before/after signatures and default values. No new
synth or root exports are proposed; the seven existing synth exports remain.
Typing needs three overload families: Definition with required inputs, str with
required source_format="qmod", and PathLike[str] with optional source_format.
The shared implementation signature does not replace those static guarantees.

| Input | Resolution and source-only options | Python-specific options |
| --- | --- | --- |
| Existing Definition | Existing capture; source_format/name/version must be None | inputs remains required semantically and statically |
| str | Always text; source_format="qmod" required; source_name defaults to `<qmod>` | inputs, input_state, constants, preparation and preparation_map must be None; functions must be empty |
| PathLike[str] | Local .qmod file; source_format may be None or "qmod"; source_name must be None and diagnostic name is the supplied filesystem path | Same source restrictions |

Resolve PathLike before callable capture; os.fspath must return str. Byte paths,
bytes, unsupported objects/formats and unknown filename suffixes are rejected.
The .qmod suffix is case-sensitive in this first profile. A str ending in .qmod
does not cause file I/O, even if such a file exists. With an explicit text
format, it is parsed as text and fails as malformed source. No content sniffing,
Python eval/exec, import from source, URL download or .fq inference occurs.

All type/format/option conflicts fail before source reads or capture. Explicit
empty inputs={} and constants={} on native input are rejected rather than
ignored. Explicit empty functions=() is indistinguishable from its default and
is allowed. A callable without inputs still raises TypeError before capture;
zero-argument Python functions continue to require inputs={}.

## Source context, file ownership and bounded work

None source_version on native input resolves to the bundled, documented profile
for upstream revision 2bfd835e73b6882fcc515a725098b04766c238e7. An explicit version
must equal that supported revision; other versions fail before file I/O. No
latest-version lookup or network request occurs. Success reports the resolved
version. It is declared compatibility context, not authentication, a Classiq
SDK version or proof of arbitrary original-file provenance. The retained corpus
continues to use its independent source/license manifest.

Ecosystem owns local file acquisition, UTF-8 decoding and adaptation. Read at
most 262,145 bytes and reject above the existing 262,144-byte cap. Consume one
bounded read from an opened regular file, verify the descriptor type, close it
on success and failure, and preserve CRLF/content without normalization. Refuse
directories, FIFOs/devices and malformed UTF-8; tests must include their early
refusal. A normal symlink may resolve to a regular file. The supplied path stays
the diagnostic name; no claim of filesystem authenticity or a race-proof
snapshot is made. File changes after capture do not silently alter or reload an
already synthesized Program.

Keep the existing parser caps: 4,096 tokens, depth 32, 78 digits per decimal
literal, unsigned widths up to 256, 4,096 expanded effects and the Compiler's
separate total-node/live-width limits. They bound source admission, not a
promise that all admitted models fit execution. ConstructionBudget still caps
candidate work; final ceilings still describe the complete selected executable.

## Execution, memory and result reading

Both source forms use the existing private frontend and register workflow,
then the same complete Program factory as Python. Synthesis plans once and
never executes. Keep exact single-batch CPU/torch/complex128, the finite two
schedules, existing logical/line targets, actual layouts and constraint repairs.
No user-provided preparation or implicit uploaded state is attached to native
source; its ordered allocate/initialize/gate operations describe preparation.

None memory_limit_bytes retains today's Python behavior. On the newly admitted
native input it resolves to a documented 512 MiB planned-state ceiling. This
format-specific default must appear in the report, never as an undisclosed
constraint. A positive plain integer overrides it for either input kind;
booleans, zero and negatives fail. None is not an unbounded native opt-out.
Explicit limits use the same Runtime options and checked state_bytes refusal
for both producers. An over-budget state plan raises existing fq.PlanningError
before the Program is admitted; this narrow change replaces the private
memory-check ValueError for the newly exposed option only. No actual state
allocation occurs in synthesis. This is not a peak-RSS/kernel-temporary budget.
The 512 MiB default admits the unchanged 23/24-qubit originals on this profile;
host capacity remains a separate practical prerequisite.

fq.run(Program), its override rejection, return type, sealed plan identity and
serialization stay unchanged. Program.probabilities keeps its exact signature
and all budgets. Default reading always means the logical name result, for
both formats. A native program without that name must select its own names;
error messages must list available names and an example selection. Do not
invent result, automatically select every output or silently discard outputs.
All declared source outputs stay live. Actual result association, normalization,
omitted-mass reporting and physical named-port checks remain authoritative.

## Report and expert-view contract

Use existing source_resources for the whole constructed circuit before target
transforms and resources for the actual checked executable. Both include all
preparation, computation and cleanup. Do not add a duplicate construction_cost
or construction_resources alias. Their scopes agree across input kinds even
when source and final costs differ.

The proposed common public report keys are resources, source_resources,
workspace_qubits, candidate, function, source_locations, scope, register_widths,
source_format, memory_limit_bytes and memory_limit_origin. The last four are
additive fields on the Python report. register_widths describes every readable
logical name, including preserved Python inputs; it contains no physical line numbers.
source_format is python or qmod, and memory_limit_bytes is the effective value.
memory_limit_origin is profile_default or explicit, so the effective ceiling
does not hide whether the caller provided it.
Retain Python's existing scope value and use that same whole-program scope for
native input. Reports and nested name/width mappings are read-only snapshots.

Native reports additionally contain source_name and source_version. Preserve all
existing Python keys and meanings, including input_widths, result_width,
input_state, constants, max_qubits and arithmetic_resources. The last remains
Python's isolated arithmetic cost; it is not a common report key. On the public
native path omit arithmetic_resources rather than label a prepared full
RegisterCandidate as pure arithmetic. Private output_widths migrates to common
register_widths; private source identities remain optional expert diagnostics.
No report serializer or new public schema version is implied.

Program.arithmetic is not among the currently admitted public properties. Keep
it an internal diagnostic field with its profile-specific representation; do
not rename a protected property, promote it as a common arithmetic-only block,
or require it in user examples. circuit, report, execution and probabilities
retain their admitted roles. This settles the public resource presentation
without introducing a second Program or breaking Python report consumers.

## Failure categories and responsibility

| Failure | Public behavior after approval | Owner |
| --- | --- | --- |
| Wrong Python type, missing Python inputs | TypeError before capture/read | Services |
| Conflicting or inapplicable options/format | ValueError before capture/read | Services |
| File missing, permission failure | Existing OSError subclass and errno | Ecosystem |
| Bad UTF-8, source budget, unknown profile, native parse/role/range error | Existing public ecosystem.InteropConversionError with its existing report and issues | Ecosystem |
| Finite candidates infeasible or search incomplete | Existing synth.SynthesisError and checked repairs | Compiler/Services |
| Planned state exceeds effective memory ceiling | Existing fq.PlanningError before admission | Services/Runtime boundary |
| Changed target, wrong result, invalid read budget or normalization | Existing checked execution/readout failure | Existing Program/reader |

Reuse InteropConversionReport/Issue for source errors instead of exporting
private HybridVerificationError or inventing another diagnostic model. Report
adapter is qmod, direction import, and framework_version is None because no SDK
version is observed. Preserve existing native diagnostic codes and complete
filename:line:column text in issue messages. Codes are structured; coordinates
are message text in this initial existing report schema. Do not add speculative
location fields to that protected schema. This does not register a new Interop
adapter or claim an InteropImportResult containing construction IR. Do not wrap
candidate SynthesisError in a conversion error or catch unrelated internal
exceptions as a successful/partial import.

Compiler remains the only construction verifier/lowerer; Runtime owns execution
and its result. Ecosystem supplies owned records immediately. Services composes
and reports; synth is still a direct re-export. Ordinary Python/Circuit use must
not eagerly import the Qmod frontend or any Classiq, SDK, network dependency.

## Implementation split and acceptance

After explicit API-owner review, one implementation PR may contain separately
owned commits against this contract:

1. Ecosystem: bounded file/text acquisition and existing structured source-error
   adaptation under _qmod; keep the existing parser/grammar and shared model.
2. Services: input dispatch in synthesis.py, reuse _synthesis_registers and
   _complete, effective memory/report fields and the unchanged named reader.
   No new root run branch or public entry factory is needed.
3. Integration: approved exact signature/contract deltas, typing overloads,
   release note, public examples and evidence metadata. Preserve historical
   snapshots; do not regenerate them as approval. Keep exports at seven.

Required acceptance for that future PR:

- Public file and explicit-text journeys use one unchanged run/read consumer;
  Python positional and function= keyword calls preserve their current behavior.
- Static typing requires inputs for Python, accepts source-only valid calls and
  refuses mixed-profile arguments where expressible. Runtime guards still cover
  all invalid combinations before I/O/capture/planning.
- Test a str that names an existing file without opening it, unknown versions
  and formats, oversized/invalid/nonregular files, CRLF, explicit source names,
  and file replacement after capture. Never execute source Python or fetch URLs.
- All three unmodified originals pass the public entry at original dimensions;
  retain separate source/parsed/lowered/executed/full-state counts. Reuse full
  complex reference, mapped clean workspace <=1e-12 and exact named marginals.
  Larger original tests remain explicitly budgeted slow/integration cases.
- Both objectives and real line routing retain nontrivial maps, coherent phases,
  source meanings, actual costs and incomplete-search refusal. Reject impossible
  memory plans and preserve existing read caps; demonstrate an explicit larger
  read on the 23/24-qubit originals.
- A preparation-only native case has real source/final costs and no public
  arithmetic-only cost. Existing Python arithmetic_resources remains unchanged.
- All public admission, serialization, lazy-import, architecture, source/license,
  capability and fresh-head CI gates pass. No optional SDK or cloud is required.

This proposal's tests exercise today's private source producer behind the
already public Program/run/read consumer and expose the cost-label difference.
They do not implement public dispatch, file acquisition safety or proposed report
fields. Those acceptance rows remain unimplemented until the next PR.

## Review, compatibility and remaining milestones

API-owner naming, overload, memory-default, error/report and compatibility review
is pending. The root run/probabilities signatures, seven exports, serialization
and old Python report values stay unchanged. The synthesize type/default delta
and additive report/error behavior are expressly proposed here. No existing
supported route is removed, so no deprecation/removal version is proposed.
The private native arithmetic_resources key had no public promise and is not
carried into the newly admitted native report.

Alternatives considered: a Source wrapper adds a public concept without a
current need; a separate run_qmod duplicates workflow ownership; strings used
as filenames introduce implicit I/O ambiguity; changing default reading to all
outputs changes cost and meaning; a new arithmetic-only public block would
require new multi-output/phase/lifetime guarantees. None is selected.

The initial source grammar is unchanged: one main, fresh unsigned scalar
outputs, supported preparation and +/*/literal square. Arrays/views, Input/Const,
free/drop, helpers and the general library remain explicitly unsupported.
Three originals do not establish all 216 files or SDK/cloud/hardware parity.
The independent developer trials remain 0/2 and release_authorized remains
false. An implementation approval does not fill those records or authorize a
release. Library expansion follows the existing dependency-ordered roadmap.
