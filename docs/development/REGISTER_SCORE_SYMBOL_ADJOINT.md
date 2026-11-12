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

Q-0747 remains open for custom lane reservations, foreign graph/dtype admission
and arbitrary WireSymbol subclass protocols. The six builtin glyph adjoints
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
