# Core component register scores

`drawer/component_score.py` projects an existing typed `ProgramRecord` into
`RegisterScore`; `register_score_data.py` supplies glyph records to the existing
JSON, offline HTML and Matplotlib renderers. Compiler owns validation and graph
selection. Drawer owns presentation only: no decomposition, tensor supply,
execution, backend selection or protected Core schema changes occur here.

## Shortest workflow

```python
from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.drawer.register_score_mpl import draw_register_score

record = addition_component_program(5)
score = component_register_score(record)
data = register_score_data(score)
figure, _ = draw_register_score(data)
figure.savefig("addition.svg")
```

Default symbols use native display names. An optional `style(function, step)`
callback can set the step's gate and control values, while preserving its exact
input/output identities. An explicit `wire_symbol(step, register, side)` callback
to `register_score_data` can supply component-specific glyphs. Its connector call
has `register=None, side=None` and must return `Text` with string `text` and
integer `fontsize`. Callback exceptions propagate. These presentation callbacks
do not infer numerical providers from function names or change execution.

## Ordering and ownership

The selected entry is a finite, one-block typed-call graph admitted by Compiler's
existing component traversal validation. Quantum words, including scalar width
zero, and owned classical bits now retain their logical display lanes. Core
validation owns executable descriptor admission. Symbolic words, conditioned
calls and parameterized calls remain outside this display profile. Empty array
dimensions retain their existing zero-port semantics. Classical integer metadata
can be laid out directly, while Core executable validation still refuses those
descriptors.

Stored block declaration order seeds layout; the existing register-score owner
computes display dependency generations and reuses free lanes. Physical component
scheduling may reorder independent calls and is intentionally not reused as a
display ordering authority. That reordering can change horizontal endpoints even
when the physical computation is equivalent.

THRU outputs keep their input's display lifetime. RIGHT outputs create fresh
collision-free lifetimes before LEFT inputs release their lanes. Shaped registers
retain row-major element order and word widths. No SSA identity or register
metadata is modified. Final returns must cover every live display lifetime once.

Single-bit Add follows the pinned direct CNOT branch with no split/join wrapper.
Larger Add uses the existing factorized graph and shared physical sequence;
drawing does not alter its measured erasure or exact adjoint.

## Locked-source evidence and limits

`tests/unit/test_drawer_component_score.py` compares unchanged Qualtran v0.7.0
at `bcb17bb25360ed68ce3bc084645db3e929799389`: atomic X/CNOT/Toffoli, all
four original closed notebook examples, and expanded Add at widths 1, 2, 3, 5,
and 32. Explicit presentation callbacks reproduce source component glyphs.
Sequential/dependency coordinates, horizontal/vertical lines and glyph attributes
match. Native identity strings and open-boundary register/dtype text differ and
are normalized explicitly in those comparisons; closed examples need no boundary
text normalization. This is not a claim of byte-identical JSON or general
source-wide drawing coverage.

| Expanded graph | Columns / vertical lines | Horizontal lifetimes | Port records | Maximum lane |
| --- | ---: | ---: | ---: | ---: |
| Add(5) | 33 | 18 | 94 | 13 |
| Add(32) | 249 | 99 | 688 | 94 |

These are presentation counts, not primitive gates or cost-model estimates. The
unchanged physical/resource accounts remain in [ADDITION_COMPONENTS.md](ADDITION_COMPONENTS.md).
Automatic production tensor-provider composition and built-in contraction planner
speed parity remain open; successful drawing does not certify either.

Run the focused test above in the locked reference environment, then the existing
arithmetic, component and drawer regressions. Evidence prefix:
`g2-component-score-`, including JSON/HTML and an inspected Add(5) SVG/PNG.

Local validation: 23 new cases, 8430 concentrated regression cases, 1479
coverage-run cases and 84 CPU/TN integration cases pass. One existing register
Graphviz rendering test is skipped because Graphviz is unavailable; the new
JSON/HTML/Matplotlib SVG path passes. Strict typing of 13 files, Black, Ruff,
Compiler scope, architecture, IR and protected API checks pass. The projection
has 95% statement/branch coverage (63 statements, 24 branches); conditioned-call
and incomplete-live-return refusals are not exercised in this coverage run. The
Add graph has 100% coverage (53 statements, 12 branches). Five source blobs
match the fixed inventory. No default full-suite rerun was required for this
slice; its previous seven recorded baseline failures remain unresolved.

## Lane reservation follow-up

The projection now accepts a native `lane_manager` and retains original Core
register groups on each display step. Final signature order is preserved for
release callbacks. Styles must retain that grouping as well as connectivity.
The shared heap supports ordered release predicates and per-element subclass
hooks through initial inputs, fresh RIGHT arrays, consumed LEFT arrays and final
outputs. See [REGISTER_SCORE_RESERVATIONS.md](REGISTER_SCORE_RESERVATIONS.md) for
source cadence, finite heap parity, failure handling and foreign protocol limits.


## Mixed storage and zero-width follow-up

Implementation `a8729432` completes the native storage-classification chain from
Core metadata through lane allocation, completed RegisterLine records and score
export. `register_line_flavor` in the existing lane owner is the shared classifier
for both interval observations and completed lifetimes. Classical zero-width
metadata retains its existing storage marker. Every logical scalar element gets
one lane, including a zero-width quantum word. A zero-length array has no lanes.
The classifier runs once before each register's element loop, as previously.

The component projection relies on Compiler's verified entry for executable type
admission. Its redundant positive-quantum-only port guard is removed. The layout
accepts built-in nonnegative widths, and invalid negative, Boolean and floating
widths fail before lane mutation. RegisterLine adds a default quantum flavor to
retain hand-authored five-argument construction. Flavor is keyed by logical
lifetime, so reusing a geometric lane after quantum consumption retains the new
classical classification. Plain layouts still synthesize quantum metadata.
The existing explicit `classical={...}` option can mark additional identities.
Automatic classical lifetimes keep their classification during export.

```python
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_measurement import measure_x_word_function
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score_data import register_score_data

from flagquantum.core.ir.program.register_types import classical_bit_type

signature = Signature((
    Register("q", quantum_bit_type(), side=RegisterSide.LEFT),
    Register("c", classical_bit_type(), side=RegisterSide.RIGHT),
))
builder = ComponentProgramBuilder(signature, (measure_x_word_function("measure"),))
result = builder.call_dict("measure", {"q": builder.inputs["q"]})
record = builder.finalize({"c": result["c"]})
data = register_score_data(component_register_score(record))
assert {line["flavor"] for line in data["hlines"]} == {
    "HLineFlavor.QUANTUM", "HLineFlavor.CLASSICAL"
}
```

The executable mixed witness in `tests/unit/test_drawer_mixed_score.py` composes
MeasureX, Parity(1) and two Discards, retaining a quantum spectator. The first
measurement output is a fresh classical lifetime. The parity output reuses a
released quantum lane with classical flavor. Every line endpoint, connector,
non-boundary glyph and typed callback event matches the unchanged source. Binding
before and after drawing is identical. The screenshot of a mixed two-element
classical array with a quantum bit was inspected, including blue/black lines.

Twenty parameter cases cover scalar, vector, matrix and two empty shapes across
quantum width zero, classical width zero, CBit and classical width three. Only
CBit and quantum words run through Core component projection. Classical integer
cases use direct layout metadata and separately verify Core's existing executable
refusal. This boundary is tested, including zero-width classical metadata. Native
port identity and open-boundary dtype spelling use the existing explicit
comparison adaptation. Complete callback sequences retain scalar and array
metadata, duplicate THRU calls and independent connector calls. JSON and embedded
sequential data retain the entire native record. Source fixture construction
subclasses Bloq only inside tests. Source drawing code remains unmodified.

The mixed-storage batch initially aligned source boundary text and source
port-record order for pixel comparison. The following traversal batch removes
that order adaptation. The renderer continues to paint supplied records in their
order. Native boundary text remains an explicit naming adaptation. General
foreign and arbitrary serialized-symbol admission remain open.

The new file has 31 cases. The two prior rejection parameters for classical and
zero-width component graphs are replaced by successful end-to-end coverage.
Focused verification passes 52 cases in 3.93 seconds. Final concentrated
verification passes 583 cases in 19.37 seconds with eight existing warnings.
Independent notebook/CPU/TN integration passes 86 cases in 25.55 seconds with two
existing warnings. Fixed-style/browser regression passes 14 cases in 20.95
seconds. No cases skip. Strict typing of six changed Python files, Black, Ruff,
Compiler scope, architecture, internal IR and protected API checks pass.

Coverage spans 463 statements and 180 branches over the four production owners.
It covers 457 statements and 173 branches, 98% combined. The shared lane owner
has all 87 statements and 26 branches covered. Component projection is 96%
combined, layout 97% and score-data extraction 99%. The remaining conditioned-call
and incomplete-return branches are reported by the coverage artifact. This batch
is a single_device_fast_path presentation change. Source blob identity remains
3863bb66cdba17c4654657e275a8485d94b8da75. Evidence uses g2-score-mixed-.

| Account | Measured mixed witness |
| --- | --- |
| Actual primitive instructions | H1, measurement1, classical_xor1 |
| Discard emission | Zero instructions |
| Source MeasureX analytical model | Measurement1, T-equivalent0, Clifford0 |
| Source Parity analytical model | Unavailable in the locked source |
| Display storage | Two initial quantum lifetimes, two classical lifetimes, maximum three concurrent lanes |
| Added by drawing | Zero physical gates, work wires or measurements |

The original MeasureX model omits its physical H and remains unchanged. Primitive
instructions and analytical values are separate. Existing forced-branch complex
amplitude/density, clean spectator, parity feedback, discard and ordinary append
regressions pass in the concentrated set. The broader measured MultiAnd witness
keeps its separately documented forward and erasure counts. The historical
full-default-suite 600-second timeout remains open. Q-0747 stays open and the
fixed ledger counts remain 30 closed, 1250 open, 386 active parents and 1187
capability records.

```bash
python -m pytest tests/unit/test_drawer_mixed_score.py \
    tests/unit/test_register_owned_x_measurement.py \
    tests/unit/test_register_classical_parity.py \
    tests/unit/test_register_classical_discard.py -q
```


## Source adjacency traversal acceptance

Implementation `0054242d` extends the existing adjacency reconstruction to both
boundary nodes. Connections retain original consumer-call order within each
neighbor group. The later connection-seeded traversal correction below places
both boundaries in source topological order. Each component retains predecessor
then successor groups. The first THRU glyph is retained while both typed callback
invocations still run. Display-identity callbacks retain their existing single
visible-port invocation and connector cadence. Geometry, portable record fields
and native identities use the existing owners. The change belongs entirely to
Drawer's single_device_fast_path presentation chain.

Ordered comparisons now check every normalized record, without sorting or
reordering source ports. Add widths 1, 2, 3, 5, 16 and 32 use ordinary and stateful
symbols. Scalar, vector, matrix and empty arrays retain complete metadata.
Independent calls, idle wires, zero calls and swapped final returns exercise
boundary neighbor grouping. Hand-authored scores without final-order metadata
use completed-line order, retaining their port mapping.

Twenty mixed-storage/shape cases compare every Agg pixel with unmodified source
record order. Only boundary label text is adapted to FlagQuantum-owned names and
dtype spelling. Long directional boxes overlap in the same order as the source.
The inspected PNG deliberately retains this source appearance. The baseline
producer from `5c7777be` fails the new ordered assertion as an expected negative
control. It runs in a separate test process without modifying tracked sources.

The existing swapped-return fixture formerly changed destination indices in an
already-built source graph, preserving its original edge insertion order.
It now builds swapped returns through upstream BloqBuilder.finalize, matching the
native finalization entry. This is an explicit fixture correction. Arbitrarily
mutated foreign connection histories remain outside native graph admission.
The first focused attempt also assumed missing final-order metadata preserved
raw order. That expectation now checks the documented mapping-preserving
completed-line fallback. Initial failures are retained in the evidence log.

The batch adds 24 parameter cases. Focused verification passes 87 in 2.92 seconds,
concentrated regression 607 in 24.27 seconds, independent integration 86 in 10.90
seconds and fixed/browser regression 14 in 16.97 seconds. No tests skip.
The concentrated and integration suites retain eight and two existing warnings.
Strict typing of seven Python files, Black, Ruff, Compiler scope, architecture,
internal IR and protected API checks pass. Four production owners cover 464 of
470 statements and 173 of 180 branches, 98% combined. The score-data owner is 99%.

Physical instruction, complex-amplitude, measurement-branch, clean-work and
ordinary append regressions pass in the concentrated suite. Presentation adds
zero gates, work bits or measurements. The prior mixed witness still emits H1,
measurement1 and classical_xor1. Its source MeasureX analytical model reports
measurement1, Clifford0 and T-equivalent0. The locked Parity model is unavailable.
Actual and analytical accounts stay separate.

Evidence uses `g2-score-order-`. The musical-score source blob remains
`3863bb66cdba17c4654657e275a8485d94b8da75`. Q-0747 remains open for general foreign
graph/dtype and arbitrary serialized-symbol admission. Core classical integer
execution and the historical full-default 600-second timeout remain separate
limits. The next batch should inspect the existing owner boundaries for foreign
admission without expanding protected schemas or adding a parallel symbol model.


## Default register-symbol projection

Implementation `7f980ce1` connects declared Core metadata to the existing default
symbol path. A caller can obtain register-aware labels with
`register_score_data(component_register_score(record))` without supplying a
callback. This follows the locked `Bloq.wire_symbol` default. Local labels use
the declaration name and row-major index, with comma-space formatting for
multidimensional indices. LEFT and RIGHT arrows retain their existing painter.
Default metadata-backed titles longer than ten characters keep eight followed
by two periods. Titles use the owned native function name, while lifecycle keys
continue to own record identity. Symbol selection consumes the existing quantum/classical flavor and geometry.

The existing explicit gate style retains control circles, addition markers and
allocation/release glyphs. Both callback forms retain precedence and invocation
cadence. Plain layouts without declarations keep their existing display-identity
labels and title behavior. Default metadata is checked through the existing
register-group owner before any records are generated. Register groups, default glyphs and lifecycle identities retain their existing
owners.

```python
from flagquantum.compiler.registers.arithmetic.addition_component import addition_component_program
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score_data import register_score_data

score = component_register_score(addition_component_program(3))
data = register_score_data(score)
assert data["soqs"] and data["vlines"]
```

The new 51-case file directly executes the unmodified upstream default method.
Reference fixture class names and native function names receive the same title.
Forty cases cover two storage kinds,
five shapes and four title lengths, including exact lengths ten and eleven.
Five cases cover same-named LEFT/RIGHT arrays with distinct opaque lifecycle
keys. Three compare every Agg pixel with only boundary text adapted and source
record order retained. Plain-layout and explicit-style compatibility plus both
malformed metadata directions complete the matrix. All-empty directional source
graphs raise the existing RightDangle KeyError, while native empty scores remain
defined. Initial fixture expectations for that exception and a strict-typing
failure from dynamic replacement keywords were corrected and retained in logs.

Focused verification passes 51 cases in 2.46 seconds. Concentrated regression
passes 658 in 37.41 seconds with eight existing warnings, independent integration
86 in 16.74 seconds with two existing warnings, and browser regression 14 in
28.02 seconds. There are zero skipped tests. Strict typing of the two changed Python files,
Black, Ruff, Compiler scope, architecture, internal IR and protected API checks
pass. Four production owners cover 468/474 statements and 175/182 branches,
98% combined. The score-data owner remains 99%. The old `bc110f91` producer fails
long-title comparison as an expected negative control. The default matrix-label
PNG was inspected and retains source overlap behavior.

The reference default method's blob is
`cef1f2f30bdc849048f50186616b4efce9877d8f` for `_infra/bloq.py`, verified against
the immutable inventory. The musical-score blob remains
`3863bb66cdba17c4654657e275a8485d94b8da75`. Evidence uses g2-score-default-.
Physical gate/operand/feedback, full complex-amplitude, all-branch, clean-work
and ordinary append regression remains passing. Display adds zero gates, work
bits or measurements. The existing measured H1/measurement1/classical_xor1
witness and source MeasureX measurement1/Clifford0/T-equivalent0 model stay
separate, with source Parity analytical cost unavailable.

This completes the generic default-symbol projection for admitted native
metadata. Q-0747 remains open for general foreign graph/dtype and arbitrary
serialized-symbol admission. Native function names and boundary
labels remain explicit naming adaptations. Source-specific overrides continue
through the existing caller-supplied style and symbol boundaries.


## Connection-seeded root traversal

Implementation `27be2062` follows the locked `_create_binst_graph` connection
insertion chain. Independent roots take their order from the first connection
that encounters them, including final output connections in signature order.
Original call ordinals retain identity. The shared private ordering helper serves
both sequential lane placement and portable symbol records, preventing the two
presentation paths from developing different traversal rules. An idle input
boundary can follow allocation roots in source topological traversal. It therefore
participates as an ordinary graph node instead of being emitted first.

The 130 new cases exercise the native ComponentProgramBuilder entry point.
Independent Prepare calls are consumed in ten root permutations or returned
directly, with widths one through three and an optional idle input. The matrix
checks 120 complete ordered geometry/glyph records, six stateful typed-callback
event sequences and four exact Agg buffers. Source methods remain unchanged.
Only boundary text is adapted for pixel comparisons. The immutable inventory
checks both composite_bloq.py and musical_score.py blobs. The old producer fails
the reverse-root ordinal comparison. Initial fixture rejection of zero-width
basis preparation and the first idle-boundary ordering failure are retained in
the evidence logs. Existing zero-width preparation admission is unchanged.

Local acceptance passes 130 focused cases in 2.21 seconds, 788 concentrated in
24.80 seconds, 86 independent integration in 10.61 seconds and 14 browser cases
in 21.28 seconds, with no skips. Concentrated and integration runs report eight
and two existing warnings respectively. Strict typing covers the three changed
Python files. Black, Ruff, Compiler scope, architecture, internal IR and public
API checks pass. Four presentation owners cover 486/492 statements and 185/192
branches, 98% combined. Test import formatting and a missing list annotation
were corrected before acceptance. Evidence uses `g2-score-roots-`.

Physical gate, operand, feedback, full complex-amplitude, measurement-branch,
clean-work, exact-adjoint and ordinary append regressions continue to pass.
Presentation adds zero physical gates, work bits or measurements. Measured
H1/measurement1/classical_xor1 remains separate from the source MeasureX model
of measurement1/Clifford0/T-equivalent0. Source Parity has no analytical model.

Native function names and boundary labels remain naming adaptations. Isolated
zero-port calls use original call order as the insertion tie-break. The source
obtains those disconnected nodes from a frozenset, so source-identical ordering
for those nodes is not claimed. General foreign graph/dtype and arbitrary
serialized-symbol admission remain open under Q-0747. Core classical integer
execution and the historical full-default timeout retain their separate status.
