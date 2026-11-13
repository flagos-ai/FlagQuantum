# Register score symbol adjoints

Drawer's `register_score_data.adjoint_wire_symbol` transforms the existing six
symbol records. The implementation follows Qualtran v0.7.0
`qualtran/drawing/musical_score.py` and the generic
`qualtran/_infra/adjoint.py` wire-symbol delegation at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.
It retains the existing tuple of symbol class and attribute mapping. This is a
`single_device_fast_path` presentation operation.

## Rules and attribute ownership

| Input symbol | Result |
| --- | --- |
| `Text`, `TextBox` | Append a dagger unless the text ends in a dagger, in which case apply the source's exact `text.strip('†')` |
| `RarrowTextBox` | `LarrowTextBox` with the same text |
| `LarrowTextBox` | `RarrowTextBox` with the same text |
| `Circle`, `ModPlus` | Same symbol and attributes |

Every call returns a fresh attribute dictionary. Font sizes and other existing
attributes survive. Caller attributes remain unchanged. Text glyphs require a
string `text` attribute. A missing attribute raises `KeyError`; a non-string
text value raises the native helper's explicit `TypeError`.

The source's stripping rule can remove leading daggers as well as trailing ones.
For example, `†S†` becomes `S`. Applying the operation twice to arbitrary text
therefore need not restore that original spelling. Tests preserve this behavior.
Directional text retains its complete spelling while the arrow switches.

## Existing callback consumers

The classical tensor-score example now obtains effect arrows by applying this
shared transform to its state arrow. Existing component-score tests compare
those states/effects with the unchanged source. Native graph inverse consumers
can supply glyphs through the same `wire_symbol` callback.

```python
from flagquantum.drawer.register_score_data import adjoint_wire_symbol

forward = ("TextBox", {"text": "S"})
inverse = adjoint_wire_symbol(*forward)
assert inverse == ("TextBox", {"text": "S†"})
assert forward == ("TextBox", {"text": "S"})
```

A callback chooses the component's forward presentation, then applies this
helper for the inverse presentation. The inverse graph and register signatures
come from the existing Compiler adjoint owner. The helper receives no graph,
physical wire or numerical state. Specialized source components can choose a
different caption policy from generic `Adjoint`, so that choice stays with the
component's callback. Generic source delegation flips register side before
requesting the forward symbol, then applies its glyph adjoint. Tests exercise
that chain for IntState, IntEffect and TGate, including connector captions.

## Executable graph and resource evidence

Tests construct real S-then-T component graphs and their exact native inverses.
Their callbacks consume the new helper. Complete score geometry and glyphs
match the corresponding unchanged source graph, under the existing explicit
native port-identity/boundary-label normalization. Physical binding before and
after drawing agrees. Primitive gates and operands match the source.

Every column of the two-qubit complex matrix is checked. An arbitrary complex
input also executes through `circuit.append(op)` and retains a zero spectator.
The broader existing measured-Add adjoint regression checks all forced
measurement histories, complex amplitudes and clean work restoration.

| Accounting layer | S/T witness |
| --- | --- |
| Actual forward primitives | S1, T1 |
| Actual inverse primitives | Tdg1, Sdg1, in reversed order |
| Source model, either direction | T1, Clifford1, measurement0 |
| Storage | Two input wires, including one spectator; source peak two |
| Added by glyph transformation | Zero physical gates, work wires, measurements or feedback |

The source QEC model and emitted primitive account remain separately checked.
The metadata transform does not perform or derive quantum inverses.

## Verification and remaining scope

The new tests cover all six symbol classes, both circle polarities, ten text
forms with three successive adjoints, dictionary ownership and invalid text.
Source blobs for musical-score records, generic adjoint delegation, S/T gates
and integer basis vectors are frozen against the inventory.

```bash
python -m pytest tests/unit/test_drawer_symbol_adjoint.py \
    tests/unit/test_drawer_component_score.py \
    tests/unit/test_component_adjoint.py -q
```

Native lane reservations and partial-line lifecycle observations are accepted
in REGISTER_SCORE_RESERVATIONS.md. Q-0747 remains open for general foreign
graph/dtype admission, arbitrary WireSymbol subclasses and source record protocols. The six builtin glyph adjoints
are now available through native records. Source-wide drawing admission and
Javascript animation remain separately tracked. Evidence prefix is
`g2-score-symbol-adjoint-` in the reference workspace.

## Local acceptance

Thirty-six new cases and 61 focused cases pass. The concentrated Drawer and
component-adjoint regression passes 302 cases in 14.45 seconds with eight
existing parser warnings. Independent native notebook kernels and CPU/TN
integration pass 86 cases in 8.62 seconds with two existing warnings. No cases
skip. The score-data owner measures 97% coverage and the example consumer 96%,
combining 147 statements and 58 branches at 97%. The new helper has no missing
statements or branches. Uncovered lines belong to existing port/control,
classical-name validation and example composition paths.

Strict typing of three Python files, Black, Ruff, scope, architecture, multi-level
IR and protected public API checks pass. All five reference blobs match the fixed
inventory. The implementation and example consumer retain their existing
ownership boundaries and require no new production dependency.


## Typed register callbacks and source adjacency order

Implementation `34553f8d` extends the existing score-data owner with the optional
`register_wire_symbol` callback. It receives the native `RegisterStep`, original
Core `Register` declaration and element index. Scalar indices are `()`, array
indices retain their row-major tuples and connector arguments are `(None, None)`.
The original register object preserves name, dtype, shape and side. Display
identity strings continue to identify lifetimes. Metadata recovery uses the
existing grouping validator and preflights every column before callback effects.

```python
from flagquantum.core.ir.program.registers import Register
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import SymbolClass, register_score_data


def symbol(
    step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    if register is None:
        return "Text", {"text": step.name, "fontsize": 10}
    return "TextBox", {"text": f"{register.name}{index}"}


score = layout_register_score(
    {"q": 1}, ("q",), (RegisterStep("X", ("q",), ("q",)),)
)
data = register_score_data(score, register_wire_symbol=symbol)
assert any(port["symb_attributes"].get("text") == "q()" for port in data["soqs"])
```

The typed callback and existing display-identity `wire_symbol` callback are
mutually exclusive. The existing form retains its invocation cadence. Callable
objects with false truth values work in either form. Callback exceptions and
invalid connector captions propagate through the existing owner.

Source `_binst_to_cxns` groups incoming connections by predecessor and outgoing
connections by successor, retaining connection encounter order within each group.
The score-data owner reconstructs that order from existing dependencies and
original operation ordinals. Layout retains final root signature order in
`RegisterScore.output_order`, including reordered returns. Split outputs can
therefore visit indices in a different order from their declaration. A typed
callback runs for all inputs, then all outputs, including the discarded second
THRU invocation. Rendering keeps the first THRU result. Connector captions follow
the source's existing multi-port condition. The ordered callback event list and
numbered glyphs are compared directly without sorting.

Layouts with display identities alone synthesize scalar opaque register metadata
using LEFT, RIGHT or THRU according to port ownership. Display names are never
parsed into array declarations. Hand-authored scores that omit `output_order`
use completed-line order for final connections. Precise root return order is
available from layout-produced scores. These are explicit native metadata
adaptations. An all-empty source custom graph raises `KeyError: RightDangle`;
the native empty graph retains its defined empty records and zero callbacks.
The original exception is reproduced in tests.

The unchanged musical-score blob is
`3863bb66cdba17c4654657e275a8485d94b8da75`. Add widths 1, 2, 3, 5, 16 and 32
compare complete geometry and callback events with pure and stateful suppliers.
Scalar/vector/matrix/empty ports, same-name opposite sides, reordered returns,
canonical metadata identity, malformed metadata, callback failures and existing
HTML/Matplotlib consumers are covered. A source test class is constructed at the
test-only foreign boundary. Production uses existing Core records and Python
containers, with no foreign runtime dependency.

Thirty-two new cases and 448 concentrated cases pass in 16.35 seconds, with eight
existing parser warnings. Independent notebook/CPU/tensor integration passes
86 cases in 8.64 seconds with two existing warnings. Both sets have zero skips.
Strict typing of three files, Black, Ruff, team scope, architecture, multi-level
IR and protected API checks pass. Score-data coverage is 98% across 143 statements
and 68 branches. Two existing builtin control-H and unknown-classical rejection
paths are uncovered by this concentrated set.

Evidence uses `g2-score-register-symbols-`. Initial adjacency-order and synthetic
return-order expectations, strict typing failures and the reproduced source
empty-graph exception are retained separately. Existing component-adjoint and
execution regressions continue to verify complex amplitudes, forced measurement
branches, clean work restoration and normal `circuit.append(op)` execution.
Presentation adds zero physical gates, work wires, measurements or feedback.
Actual primitives and source analytical estimates retain their separate accounts
in MUSICAL_SCORE_SOURCE_AUDIT.md. Q-0747 stays open for general foreign
admission and record serialization boundaries. Q-0748's original JSON asset is
tracked independently.

```bash
python -m pytest tests/unit/test_drawer_register_symbols.py -q
```


## Replacing native port painting

Implementation `1247d5fe` exposes the existing builtin painting function as
`draw_register_score_port`. The sequential renderer accepts an optional
`port_drawer(ax, port)` callable. The renderer invokes it once for each port in
record order, including boundary labels, after horizontal lines and connectors.
The supplied ScorePort retains its original identity, attributes and coordinates.
Its y value denotes a positive row. A painter uses `-port["y"]` for Matplotlib
coordinates and treats the borrowed record as read-only. The painter can delegate
ordinary glyphs to the same builtin owner. Explicit builtin replacement and the
default renderer produce identical pixels for all six classes and both Circle
polarities. Callable objects with false truth values are supported.

```python
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import ScorePort, register_score_data
from flagquantum.drawer.register_score_mpl import (
    draw_register_score,
    draw_register_score_port,
)


def paint(ax: Axes, port: ScorePort) -> None:
    if port["symb_cls"] == "TextBox":
        ax.text(
            port["seq_x"],
            -port["y"],
            str(port["symb_attributes"]["text"]),
            ha="center",
            va="center",
            rotation=15,
            bbox={"boxstyle": "round4", "fc": "lightyellow", "ec": "purple"},
        )
    else:
        draw_register_score_port(ax, port)


score = layout_register_score(
    {"q": 1}, ("q",), (RegisterStep("R", ("q",), ("q",)),)
)
data = register_score_data(score)
figure, axes = draw_register_score(data, port_drawer=paint)
assert axes.figure is figure
plt.close(figure)
```

This is a `single_device_fast_path` presentation extension. The source reference
constructs a caller-defined WireSymbol subclass and executes the unchanged
`draw_musical_score` dispatch. Its custom `draw` method paints a rotated badge.
Its `json_dict` exports a TextBox fallback. The native painter independently
constructs the same artists, delegates ordinary labels/markers to the shared
builtin owner and matches every Agg pixel at default, width-limited and
height-limited sizes. Captions, classical lines, axis limits, return ownership
and exact custom draw coordinates also match. The completed PNG was inspected.

The custom painter remains a local rendering argument. The six-class portable
JSON representation carries its builtin fallback. Reopening exported JSON or
HTML uses that fallback. The caller supplies its painter again for the custom
Matplotlib appearance. Foreign symbol objects and arbitrary serialized class
names retain their separate admission boundary. Source default WireSymbol
adjoint returns the custom instance itself, which the test preserves. Custom
inverse presentation stays supplier-owned alongside the existing builtin glyph
adjoint helper. Connector labels retain the existing Text drawing path.

Painter exceptions stop further port calls, close the incomplete native figure
and propagate the same exception object. A new rendering call retries with fresh
figure state. The original source propagates the exception and retains its
unfinished figure. Both behaviors execute in the failure test, with source
figures explicitly cleaned up by that test. This cleanup is a native ownership
adaptation. Empty native records invoke no port painter and return a caller-owned
figure. Renderers depend on the existing score records and optional Matplotlib.

Fifteen new cases pass. The concentrated regression passes 503 cases in
16.72 seconds with eight existing parser warnings. Independent notebook/CPU/TN
integration passes 86 cases in 9.14 seconds with two existing warnings. No cases
skip. Strict typing of two files, Black, Ruff, scope, architecture, multi-level
IR and protected API checks pass. All 60 statements and 18 branches in the
Matplotlib score owner are covered. The original musical-score blob remains
`3863bb66cdba17c4654657e275a8485d94b8da75`. Evidence uses
`g2-score-custom-symbols-`. The first strict check identified four unannotated
Agg method calls in tests. The test-only optional backend boundary is now explicit,
and the initial failure log is retained. Production callbacks retain precise
Axes and ScorePort types.

Existing exact-adjoint and execution regressions preserve complex amplitudes,
forced measurement branches, work restoration and normal `circuit.append(op)`
use. Added physical gates, work, measurements and feedback are zero. Actual
primitive counts and source analytical estimates retain their separate accounts
in MUSICAL_SCORE_SOURCE_AUDIT.md. Q-0747 stays open for general foreign
admission, arbitrary serialized symbol classes and custom connector-label draw
dispatch. Q-0748's original JSON asset remains independently open.

```bash
python -m pytest tests/unit/test_drawer_custom_symbols.py -q
```
