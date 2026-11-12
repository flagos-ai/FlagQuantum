# Register score browser animation

The existing `drawer/register_score_html.py` owns an offline SVG presentation
of the shared `MusicalScoreData` records. Compiler retains graph semantics and
Simulation retains numerical execution. This is a `single_device_fast_path`
presentation change. Circuit construction continues through `circuit.append(op)`.

```python
from pathlib import Path

from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.drawer.register_score_html import export_register_score_html

program = addition_component_program(3)
score = register_score_data(component_register_score(program))
Path("addition.html").write_text(export_register_score_html(score), encoding="utf-8")
```

Open the file in a browser and switch Column view between Sequential and
Dependency. Existing SVG nodes move over 750 milliseconds with cubic-in-out
interpolation. Horizontal lines grow or collapse at their starting column.
Vertical lines grow or collapse at their lower endpoint. Circles enter and exit
at screen y zero, ModPlus markers at the lower baseline. Text glyphs fade.
Changes resume from the current geometry when interrupted. An exiting key can
reenter before removal and retain the same DOM node. Separate maps own line,
caption and glyph lifetimes. Glyph identity includes its class and logical port
identity, matching the source's separate joins for each class.

## Frozen source ownership

Qualtran v0.7.0 is pinned at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Asset | Git blob | Native disposition |
| --- | --- | --- |
| `qualtran/drawing/musical_score.html` | `8731aa90a752d4fd252ea931f0527266da0c3934` | Existing self-contained HTML template and safe embedded records |
| `qualtran/drawing/musical_score.js` | `ead9b24fc77b0a772e05f18280a734ddf65ee3a1` | Existing SVG glyph renderer, keyed joins and shared animation driver |

The HTML imports D3 v7 from a CDN, then the sibling script, and fetches a sibling
JSON example. It declares manual x/y domains and a `content` container. Native
export embeds data and drawing code, sizes its viewport from the records, and
uses the existing `score-container` and selector. Embedded labels escape `<` in
the JSON script and reach SVG through `textContent`. Caller-owned file export
propagates Python serialization and filesystem errors through their existing
boundaries. The optional Score JSON URL form follows the source fetch/error
chain through
`loadScore` and `showScoreError`. The embedded viewer opens without a fetch.
HTTP loading occurs when the user submits the form.

Every source JavaScript function is accounted for below.

| Source function | Native owner and disposition |
| --- | --- |
| `makeCanvas` | Existing SVG creation and viewport sizing in `register_score_html.py`. Source numeric axes, fixed domains and margins remain a display difference. |
| `drawVlines` | `join(vertical, ...)` preserves x keys, lower-endpoint growth, movement, collapse and delayed removal. |
| `drawHlines` | `join(horizontal, ...)` preserves lane/start keys, growth, movement, collapse and delayed removal. Native quantum/classical coloring is retained. |
| `drawCircles` | `portGlyph`, `updatePort` and `join(ports, ...)` preserve filled/open polarity, radius five, motion and opacity. |
| `drawModPlus` | The same owners preserve radius seven, plus strokes, lower-baseline entry/exit and motion. |
| `drawAnyTextBox` | Shared text-glyph creation, text updates, keyed movement and fading. Click diagnostics log the event and current bound record, including after an update. |
| `drawTextBox` | Existing white, black-outlined box, half-height fourteen. Native width grows for long text. |
| `drawRarrowTextBox` | Existing right-arrow polygon, half-height fourteen, with native adaptive width. |
| `drawLarrowTextBox` | Existing left-arrow polygon with the same sizing boundary. |
| `drawText` | Existing text-only glyph. Native explicit font size is retained. |
| `musicalScore` | `render` dispatches all shared joins and one 750 ms cubic-in-out animation clock. |
| `showError` | `showScoreError` appends a red heading and details through text nodes. Repeated failures retain their history as in the source. |
| `make_from_data` | `loadScore` uses fetch, checks HTTP status, parses JSON, and invokes the existing renderer. The URL form exposes that path alongside embedded records. Structural validation and newest-request ownership are native adaptations. |

Source globals comprise the D3 linear-scale constructors and axes, circle radius,
margins/insets, box half-width/height, domains, derived dimensions/ranges, canvas
and data group. Native code retains glyph radii and box height, owns responsive
coordinate functions and shares one persistent SVG. D3 selection/data joins are
implemented by local maps and DOM operations. `requestAnimationFrame` drives the
shared easing function. Production adds zero dependencies. Reference tests use
D3 7.9.0 with SHA256
`f2094bbf6141b359722c4fe454eb6c4b0f0e42cc10cc7af921fc158fceb86539`.
Source copyright attribution remains on the production owner.

## Browser acceptance and remaining boundaries

The reference test embeds the frozen JavaScript unchanged in real Chrome with
the pinned D3 bundle. Both pages use a deterministic clock. Reference mutable
scale ranges are set explicitly to the native test viewport before comparison.
This establishes timing and geometry parity in that coordinate system. It does
not establish whole-page pixel parity.

All six glyph classes and both line classes are sampled at zero, one eighth,
one quarter, one half, three quarters and one full duration. Enter/update/exit,
polarity/text changes, added/removed keys, a class change on an existing identity,
and interrupted exit/reentry are compared. Normal selector switching independently
checks retained DOM identity, intermediate geometry, immediate interruption and
final sequential/dependency coordinates. Further cases cover an empty page,
long labels, explicit font size, captions, classical color and script-like label
text. Existing current/legacy JSON browser checks pass as well.

The source's numeric axes, light-blue horizontal lines, fixed 44-pixel box width,
and default text size remain explicit open differences. Native captions and the
sequential/dependency selector are
existing presentation features. Q-0744 and Q-0746 remain open pending disposition
of those remaining whole-file behaviors. Generic foreign graph and line-manager
protocols remain owned by Q-0747.

Run the browser checks with Chrome available and the locked reference environment
on the Python import path.

```bash
export FLAGQUANTUM_CHROME='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
python -m pytest tests/integration/test_drawer_score_animation.py \
  tests/integration/test_drawer_register_score_browser.py -q
```

Expected local result is six passes with no skips. Local acceptance on
2026-11-13 uses implementation `11f559cd` and includes four new browser cases, six browser passes in 3.77 seconds,
343 concentrated passes in 15.58 seconds and 86 independent notebook/CPU/tensor
passes in 9.30 seconds. Existing parser warnings number eight and existing
integration warnings number two. Python coverage for `register_score_html.py`
is 46/46 statements and 12/14 branches, 97% combined. Browser JavaScript behavior
is exercised by the real-DOM assertions above and has no instrumented percentage.
Strict typing of both changed Python files, Black, Ruff, Compiler scope,
architecture, internal IR and protected API checks pass. Evidence prefix is
`g2-score-animation-` in the external reference environment. The executable
Add(3) example exports successfully. Its settled-frame Chrome screenshot was
visually inspected using the same deterministic clock as the reference tests.
A direct virtual-time screenshot captured an incomplete entry frame and is
retained separately from the settled-frame evidence.

## JSON URL loading

Export the viewer and JSON through the existing owners, then serve their output
directory with the standard-library HTTP server.

```python
from flagquantum.drawer.register_score_data import export_register_score_json

Path("addition.json").write_text(export_register_score_json(score), encoding="utf-8")
```

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Open `addition.html` on that server, enter `addition.json` in Score JSON URL and
press Load score. The same SVG joins update to the fetched records. Relative and
absolute URLs use browser fetch semantics. HTTP status failures, invalid JSON,
CORS denial and local-file fetch refusal follow the frozen source's error text,
details and red heading. An error appends a new diagnostic, preserving previous
errors across retries. Error text and imported labels reach the DOM through text
nodes. Clicks on each text-glyph class log the event and current bound record.
Circles and ModPlus markers retain their source behavior.

The imported JSON retains all original fields and identities. Loading accepts the
current structured labels and legacy string-label records. Core graph validation
and numerical execution remain with their existing owners. The browser checks
only the dimensions, consumed coordinates, admitted glyph attributes, labels and
unique join keys before replacing the current view. Twenty-four malformed
record variants are refused before the original view changes. Duplicate glyph
or line keys are outside this import surface. Native JSON export supplies unique
sequential keys.

Two adaptations preserve browser state. A failed import keeps both existing
views and their current selection. The latest requested URL owns the next
update, so older responses and older errors leave the current display intact.
The source itself has no request-generation guard. Imported source JSON provides
one sequential layout, while the native embedded viewer contains both sequential
and collision-free dependency layouts. After a successful import the selector
shows Sequential and disables Dependency, and the stale dependency records are
removed. Reloading an exported viewer supplies both computed layouts again.

`test_drawer_score_loading.py` executes ten new cases against unchanged source
JavaScript and the pinned D3 bundle in real Chrome. They cover current/legacy
exports, a directly serialized locked MultiAnd, 404, malformed JSON, cross-origin
refusal, local-file refusal, transactional admission, successful retry, safe label
text and out-of-order requests. Success cases compare complete glyph/line
snapshots after explicitly aligning mutable source scale ranges. Direct MultiAnd
records are compared without translation. Default long-label spacing remains
an open style difference. The initial direct-source check retained a failure
showing native x=224 versus source x=110 at the same logical column. The aligned
rerun preserves every geometry assertion and configures the reference scale
range explicitly. Error cases compare complete heading/detail text and computed
color. Native ownership/admission adaptations have independent assertions.

Local acceptance on 2026-11-13 uses implementation `5fdd4bda` and has ten new
cases, sixteen browser passes in
15.52 seconds, 343 concentrated passes in 14.93 seconds and 86 independent
notebook/CPU/tensor passes in 8.86 seconds, with zero skips. Existing concentrated
and integration warnings number eight and two respectively. Strict typing of
the two changed Python files, Black, Ruff, Compiler scope, architecture,
internal IR and protected API checks pass. Exporter coverage remains 46/46
statements and 12/14 branches, 97% combined. A real-HTTP, form-triggered error-panel
screenshot with the retained MultiAnd graph has been visually inspected. Its
settled graph uses the deterministic test clock.

Evidence prefix for this batch is `g2-score-loading-`. Browser clocks are
controlled for deterministic completion. Python coverage measures the exporter
and dependency projection, while JavaScript behavior is checked through DOM
assertions without an instrumented coverage percentage.

## Separate resource accounts

Existing physical operand-order, full-complex-amplitude, forced measurement
history, work restoration and ordinary append tests pass in the regression set.
The browser renderer emits presentation records only.

| Account | Existing negative-control MultiAnd consumer |
| --- | --- |
| Actual forward primitives | H6, T6, Tdg6, CX18, S3, X2 |
| Actual erasure primitives | H3, measurement3, conditional CZ3, reset3, X2 |
| Source forward model | AND3 and Clifford2, equivalent T count 12 |
| Source adjoint model | Clifford5 and measurement3 |
| Physical storage | Four controls and three work/output wires, source peak seven |
| Added by animation | Zero primitives, physical work wires, measurements or feedback |
