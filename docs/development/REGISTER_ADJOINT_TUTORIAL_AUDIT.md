# Complete register adjoint tutorial

Q-0138 maps the complete frozen `qualtran/Adjoint.ipynb` at Qualtran v0.7.0
revision `bcb17bb25360ed68ce3bc084645db3e929799389`. Its Git blob is
`d063fb12c9648da04de773dae977cec23973f004`. The native notebook preserves all
17 cells and their kinds, with explicit source-cell metadata. The source stays
unchanged. The Apache attribution is retained in the native helper and notebook.

## Ownership and complete mapping

| Source cells | Native behavior and owner |
| --- | --- |
| 0–1 | Explain adjoints and draw zero state/effect through Core directional registers, Compiler tensor-root expansion and Drawer graphs |
| 2–4 | Construct the four-bit GHZ preparation, reverse the hierarchy and execute preparation followed by its adjoint using `Circuit.append` |
| 5–7 | Compare exact X/CNOT self-adjoints and distinguish the specialized AND compute and measurement-assisted erasure implementations |
| 8–11 | Construct and reverse the three serial atoms, then draw both complete call graphs |
| 12–15 | Store H followed by T, derive T-adjoint followed by H, draw both graphs and verify their complete complex matrices |
| 16 | Explain decomposition-based state/effect semantics and the physical clean-release precondition |

`examples/algorithms/adjoint_programs.py` composes existing owners. Core owns
register types, state/effect directions and immutable functions. Compiler owns
hierarchical adjoints, physical binding and explicit exact leaf mappings.
Simulation owns tensors and statevectors. Drawer owns graph rendering. The
shared `FieldOperation` example adapter retains the `circuit.append(op)` entry.
The native tutorial executes with imports from Qualtran, Cirq and Kickmix denied.
IPython and Graphviz are optional presentation dependencies.

The source's `q` basis-function port becomes the native `val` port. The comparison
normalizes this declared spelling and graph instance labels, then checks every
call connection, shape and operand in all three forward/inverse graphs. The
source serial atoms have explicit X tensors. The native examples use those X
bodies and explicitly self-adjoint leaves. These examples do not implement an
automatic generic wrapper protocol.

## Numerical, lifetime and resource evidence

The whole source and native notebooks execute. The complete GHZ preparation
and effect tensors match without phase normalization. The H/T forward and
inverse tensors match, including application to a complex entangled spectator.
The emitted gate operands and order match the unchanged source decomposition.
AND compute and erasure match the source's primitive stream and feedback.
Both measurement outcomes recover the full complex input with amplitude factor
`1/sqrt(2)` and exact zero support on the target workspace.

| Case | Emitted primitives and ownership | Independent source model |
| --- | --- | --- |
| GHZ preparation | H, CX, CX, CX; four allocated output bits | No extra model is asserted by this tutorial |
| GHZ inverse | CX, CX, CX, H; four released bits on the promised GHZ input | No extra model is asserted by this tutorial |
| Phase preparation/inverse | H,T / T-adjoint,H; zero workspace and measurements | One T, one Clifford, zero measurements |
| Three serial atoms | Three X instructions; zero workspace and measurements | Forward source synthetic estimate is 300 T |
| AND compute | 13 primitives; one target workspace | One AND block, equivalent to four T |
| AND erasure | Four primitives including measurement, conditional CZ and reset | One Clifford and one measurement |

Preparation bits are returned outputs. The inverse can physically release them
only when the input satisfies its clean-release promise. Its tensor is the GHZ
effect, which also has a mathematical action on arbitrary inputs. This distinction
is retained explicitly in the teaching material.

The unchanged source's cost query for the serial example's default adjoint raises
`DecomposeTypeError` because `TestAtom` is atomic and its static forward cost is
not delegated by that wrapper. The test records the exact exception. No inverse
300-T source result is invented. Full automatic diagram-exponent/name delegation
and other generic wrapper protocols remain under Q-0147/Q-0148.

## Acceptance scope

The nine original adjoint tests, including the real Jupyter notebook execution,
pass unchanged. Native acceptance covers the entire tutorial, exact call graphs,
matrices, primitive streams, feedback, clean work and eight rendered SVG graphs. The GHZ and phase inverse PNG renders were
visually inspected.
The concentrated regression passes 1352 cases and the integration regression
passes 96 cases. Strict typing passes for the helper, test and all extracted
notebook code. Formatting, architecture, scope and IR check results are stored
under `ftqc-reference-env/g2-adjoint-tutorial-*`. The dependency scan retains 689 whole-repository findings with zero findings
in this batch. The default smoke/unit command stops at the existing public
docstring registry failure after 434 passed and 51 skipped. This is recorded
in `g2-adjoint-tutorial-default-first-failure.log`, and the full default suite
is not claimed green.
Q-0138 can be closed on this complete tutorial scope. Q-0147/Q-0148 remain open.
