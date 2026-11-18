# Complete phase-estimation package mapping

Frozen `phase_estimation/__init__.py`, blob
`3f3e49797e3358ec6c581fac78b3a5bc75c9cd97`, at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389` contains five imports. Each maps
to an existing Algorithms primitive owner through its direct native module.
The package mapping closes Q-0619.

| Frozen package entry | FlagQuantum owner under `algorithms/primitives` | Accepted evidence |
| --- | --- | --- |
| KaiserWindowState | `kaiser_window.KaiserWindowState` | `KAISER_WINDOW_FILE_AUDIT.md` and `KAISER_WINDOW_NOTEBOOK_AUDIT.md` |
| LPResourceState | `luis_perina_state.LuisPerinaState` | `LUIS_PERINA_STATE_FILE_AUDIT.md` and `LUIS_PERINA_NOTEBOOK_AUDIT.md` |
| RectangularWindowState | `phase_estimation_window.RectangularWindowState` | `PHASE_ESTIMATION_WINDOW_FILE_AUDIT.md` |
| QubitizationQPE | `qubitization_phase_estimation.QubitizationPhaseEstimation` | `QUBITIZATION_PHASE_ESTIMATION_FILE_AUDIT.md` and both complete QPE notebooks |
| TextbookQPE | `textbook_phase_estimation.TextbookPhaseEstimation` | `TEXTBOOK_PHASE_ESTIMATION_FILE_AUDIT.md` and its complete notebook |

Import mapping preserves all five original package identities and exact source
module destinations. The native constructors belong to their individual domain
modules. All three window entries preserve complete registers, output ownership
and phase width. The imported Kaiser constructor preserves the original exact
coefficient array. Physical consumers use existing named PREPARE and QPE
bindings, which return operations for `circuit.append(op)`.

The associated complete-file acceptance is reused. Concentrated regression runs
all five component families, all five phase-estimation notebooks, the independent
quantum-walk ordering, prepared-state reflection and QFT dependencies. It covers
gate operands and parameters, exact complex states, applicable exhaustive
measurement histories, workspace cleanup, execution replacement and independent
resource queries. Public API checks use the repository's protected snapshot.

## Retained component boundaries

Q-0622 retains Kaiser source decomposition and ideal-versus-quantized boundaries.
Q-0631 retains large physical factory admission, SDK-default leaf synthesis,
original paper Fig. 2 differences and the separately observed historical
Clifford accounting discrepancy. Q-0634 retains generic controlled-Bloq leaf
and walk/GQSP admission boundaries. Existing source failures, strict skips and
xfails remain in those whole-file audits. Package import closure preserves
these component-specific acceptance obligations.

## Validation

The complete concentrated run passed 278 tests. Three original notebook/IPython
warnings remain visible in `g2-phase-package-regression.log`. Black, Ruff, strict
mypy for the changed test, Algorithms scope, architecture, multi-level IR and
protected API checks all passed. Package mapping adds no physical instructions
and reuses the component audits' measured primitives and separate model counts.
