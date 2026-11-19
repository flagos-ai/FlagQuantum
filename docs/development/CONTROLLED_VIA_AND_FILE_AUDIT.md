# ControlledViaAnd source file audit

The reference is Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. The tests verify these Git blobs.

| Source | Blob | Scope |
| --- | --- | --- |
| `controlled_via_and.ipynb` | `825c66f74d650124a0db19f21b433d680b8d4072` | Q-0549, all 14 cells and both original fixtures, including nested control |
| `controlled_via_and.py` | `3947ddf86be37f4ff3ba58fcc316a8a64922f84f` | Q-0550, grouped signature, graphs, predicate execution, attachment and resource model |
| `controlled_via_and_test.py` | `9e60e3eccb50cfdbcac9054f9471f107062c6d7a` | Q-0551, original examples, three seeded tensor cases, nested integer groups and nested X |

The compiler prepends ordered control groups and derives names from the child
signature. The predicate computes into private work bits, calls the controlled
body, and erases the predicate with measurement and feedback. The controls and
target return through the original ports. Integer groups expand in source
big-endian order using two's-complement for signed words. The nested extension
puts the new groups first and computes a single predicate. Existing component
extension tests cover attachment, name collisions, and returned binding order.

The [notebook](../../examples/notebooks/controlled_via_and.ipynb) displays both
original signatures and call graphs and binds the nested example through
`circuit.append(op)`. Tests compare ordered primitive names, operands and
feedback against the locked source, including the 64-wire integer fixture.
The four- and six-control H cases cover all 8 and 32 histories with full complex
amplitudes and zero work bits. The typed nested X case checks all eight histories
with the same output and work-bit contract. The original source suite passes
all ten collected cases.

| H fixture | Emitted T/T† combined | Emitted H-controlled primitive | Erasure measurements | Modeled And | Modeled peak qubits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four bit controls | 12 | 1 | 3 | 3 | 8 |
| Grouped integer controls | 124 | 1 | 31 | 31 | 64 |
| Six nested bit controls | 20 | 1 | 5 | 5 | 12 |

The source QEC model counts the controlled-H child as four Cliffords and two
rotations. The nested X source model matches three And calls, four Cliffords,
and three measurements.

## Complete definition and dependency map

The twelve source definitions are accounted for below. The wrapper is native
composition over existing Program records, without a foreign Bloq hierarchy.

| Source definition | Native owner and evidence |
| --- | --- |
| `ControlledViaAnd`, `__attrs_post_init__` | `quantum_control_signature`, `validate_control_ports`; quantum metadata remains separate from executable storage admission. |
| `make_ctrl_system` | `extend_component_control_system` and its actual returned adder tests. |
| `decompose_bloq`, `build_composite_bloq` | `control_component_via_predicate`; actual callback between explicit predicate compute and erase. |
| `_is_single_bit_control`, `_single_control_value` | Total encoded width and `ControlRegister.bits`; scalar/singleton and both polarities. |
| `adjoint` | Existing exact component adjoint and explicit inverse leaf bindings; all forward/inverse measurement-history pairs. |
| `build_call_graph` | Existing predicate-controlled resource queries, both QEC profiles and independent private peak accounting. |
| `get_ctrl_system` | New groups precede old groups; returned callback preserves source pop/zip behavior and invokes one merged predicate. |
| `_controlled_via_and_qbits`, `_controlled_via_and_ints` | Both complete original fixtures and the fourteen-cell native tutorial. |

The integer storage dependency now admits concrete canonical ones'-complement,
bounded unsigned and Montgomery words alongside previously accepted signed
words. Core retains dtype and parameters through ownership and rebindings.
Compiler flattens the existing bit codecs and reconstructs the original ports.
There is no conversion from an arithmetic integer into Montgomery representation
in this wrapper. Its activation value is already the stored representation.

The predicate dependency now retains Partition inverse, Split, conjunction,
Join and Partition as actual typed calls. MultiAnd contains source-ordered And
calls. The tutorial consumes that shared implementation. Original root and
ladder connection traces match in both directions. Physical primitive order,
operands, feedback and allocation lifetimes remain independently checked.

All twelve definitions and their direct dependencies now pass the complete
source-file audit, including unknown-activation graph composition and the
actual returned extension callback.
The source default refuses non-THRU child decompositions. A specialized child
with incompatible renamed directional output order fails source finalization;
the native wrapper diagnoses this before the callback. These failures are
preserved witnesses in `test_left_only_control_fallback.py`. The source also
fails singleton shape `(1, 1)` decomposition although its direct tensor works;
that distinction is retained in `SINGLE_BIT_CONTROL_METADATA_AUDIT.md`.
Unknown binary activation retains two explicit directional conjunction
declarations around the unchanged positive child callback. The existing
FunctionRecord declaration contract supplies their graph identity and ports.
Callers bind resource identities and numerical actions explicitly. Replacing
both declarations with concrete conjunctions preserves the outer consumers,
exact primitive stream, all four binary measurement histories and clean work.

For concrete widths of at least three, the source MultiAnd constructor turns
HasLength into positive controls. Native composition follows this exact rule,
including mixed known/unknown groups through the actual extension callback.
The one-bit unknown activation and symbolic-width composite construction retain
source refusals. The binary symbolic And tensor and physical circuit disagree
upstream, as documented by the predicate audit and an executable witness.

The current receipts are `g2-integer-storage-resources.json`,
`g2-integer-storage-regression.log`, `g2-predicate-root-traces.log` and
`g2-integer-storage-original.log` in the reference workspace. They record
observed primitive counts separately from all source model categories.
The changed storage widths emit zero gates. Small controlled fixtures enumerate
all measurement histories, check complete complex spectator states and clear
all work. Larger original fixtures retain exact streams and resource evidence
without claiming exhaustive statevector enumeration.

The symbolic graph batch passes 977 concentrated cases and 96 SDK integration
cases. Strict typing, Black, Ruff, scope, architecture and IR checks pass.
Whole-repository dependency diagnostics retain 687 existing findings, with zero
in changed paths. Earlier complete source suites and unchanged numerical
fixtures remain the evidence for their original ranges. The new resource
receipt separates actual resolved primitives from source model estimates.
