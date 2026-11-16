# Unary QROM source audit

The four frozen files in this batch match the Qualtran v0.7.0 inventory
at `bcb17bb25360ed68ce3bc084645db3e929799389`.

| Inventory | Native evidence |
| --- | --- |
| Q-0479 tutorial | The native notebook preserves all four examples, their source datasets and widths, the signatures and the five-row direct call graph. Its executable example uses the append protocol with a nonzero XOR target. |
| Q-0480 implementation | `source_qrom_ir` retains the source traversal. `source_qrom_calls` exposes immediate calls with literal controls and zero-selection constants. `qrom_wire_symbol` supplies the source register labels to the native score renderer. |
| Q-0481 base | `QromTable` owns concrete data, unknown-data shapes, replacement, signature and binding. `lookup_classical_values` owns the per-address classical callback and respects the original control and XOR behavior. |
| Q-0482 tests | The unchanged source suite passed 45 circuit, cost and metadata tests plus the notebook test in a separate kernel. The source reports three strict-type skips and three expected failures for its unenforced count checks. Native parity reuses the direct operand, feedback, branch, signature, classical and resource tests. |

The direct call declaration shares the bounded count traversal with
the concrete resource model. The calls retain negative AND
`(1,0)`, positive AND and adjoint `(1,1)`, MultiAnd operand lists
and the constant and width of zero-selection XOR. The comparison
reads the exact depth-one edges of the frozen QROM, including
zero multiplicities, with the source Split/Join generalizer. It
covers zero through four controls, zero selection, uniform data,
nonuniform one- and two-axis tables, and shaped outputs.

The five-row tutorial starts each address with a nonzero target and
complex amplitude. The emitted circuit acts on six port wires and
two caller-owned work wires. It uses 12 T-family gates, 26 CX and
three measurements. The source direct graph declares two X, three
negative AND, eight CNOT and three positive AND adjoints. The
separate abstract QEC model reports three AND, 13 Cliffords and
three measurements. These are distinct levels of the same call
chain. The executable tutorial checks the complete output vector
and zero work on sampled histories. Existing gate and branch tests
cover exhaustive small histories, nonzero inputs and feedback.

## Open source boundaries

- The Q-0480 implementation retains the locked emitter's failure on
  nonuniform mixed singleton axes and zero-width target words.
  Pruned out-of-range addresses have source-dependent aliasing.
  The native diagram renders the source port glyphs and uses its
  own layout, so Cirq's variable-spacing ASCII strings have no
  literal native counterpart.
- For Q-0481, the native layout validates shapes early. The source
  constructor can defer shape mismatches to its cached shape
  property, and an empty selection prefix acts as a sentinel
  in its dataset consistency check. Both shape and executable
  admission distinctions are retained in the evidence.
- The source strict-type skips compare opaque QAny targets with
  QUInt inputs at controlled XorK. The recovered notebook test
  passed once its kernel could bind local sockets.

The concentrated native suite passed 256 cases, and both lookup notebooks
passed in independent native kernels. Strict typing, formatting, team
scope, architecture, IR contracts and the public API snapshot passed.

See [the gate and branch audit](REGISTER_SOURCE_QROM.md).
