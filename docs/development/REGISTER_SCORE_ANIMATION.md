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
boundaries. Source HTTP/CORS failures and the source error panel remain outside
the native offline loader. A separate fetch API has not been introduced.

Every source JavaScript function is accounted for below.

| Source function | Native owner and disposition |
| --- | --- |
| `makeCanvas` | Existing SVG creation and viewport sizing in `register_score_html.py`. Source numeric axes, fixed domains and margins remain a display difference. |
| `drawVlines` | `join(vertical, ...)` preserves x keys, lower-endpoint growth, movement, collapse and delayed removal. |
| `drawHlines` | `join(horizontal, ...)` preserves lane/start keys, growth, movement, collapse and delayed removal. Native quantum/classical coloring is retained. |
| `drawCircles` | `portGlyph`, `updatePort` and `join(ports, ...)` preserve filled/open polarity, radius five, motion and opacity. |
| `drawModPlus` | The same owners preserve radius seven, plus strokes, lower-baseline entry/exit and motion. |
| `drawAnyTextBox` | Shared text-glyph creation, text updates, keyed movement and fading. The source's click-to-console diagnostic remains unported. |
| `drawTextBox` | Existing white, black-outlined box, half-height fourteen. Native width grows for long text. |
| `drawRarrowTextBox` | Existing right-arrow polygon, half-height fourteen, with native adaptive width. |
| `drawLarrowTextBox` | Existing left-arrow polygon with the same sizing boundary. |
| `drawText` | Existing text-only glyph. Native explicit font size is retained. |
| `musicalScore` | `render` dispatches all shared joins and one 750 ms cubic-in-out animation clock. |
| `showError` | Source red heading/details for fetch failures remain outside the embedded-data path. |
| `make_from_data` | Source asynchronous JSON fetch is replaced at the existing export boundary by embedded typed records. The HTTP loader and its failure behavior remain open. |

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
default text size, click logging and HTTP/error-panel behavior remain explicit
open differences. Native captions and the sequential/dependency selector are
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
