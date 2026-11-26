# QREF resource interchange source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The notebook and export tests
use QREF 0.11.0 and Bartiq 0.12.1 from an isolated reference environment.

| Source file | Frozen Git blob | Native disposition |
| --- | --- | --- |
| `qref_interop/__init__.py` | `741991b9f3dce077089729e3cd75efdf904c56ee` | Compiler formats owns the optional QREF entry points |
| `qref_interop/_bloq_to_qref.py` | `aab3422ba170c0c909ee152d46537f78335829b8` | Native leaf, indexed component and logical call-graph exporters |
| `qref_interop/_bloq_to_qref_test.py` | `e7092a50ce00daf63488a419b0bd8609e937fc38` | All 28 frozen cases pass. Paired tests cover the admitted native paths |
| `qref_interop/bartiq_demo.ipynb` | `60e8055c7669d4dd69c8d42c4b882c9a7bfc7fc0` | Native `resource_interchange.ipynb` covers all 38 cells by topic and runs all 15 code cells in an independent kernel |

The optional format adapter receives native register declarations, typed
connections and modeled call graphs. It splits through declarations into
input and output ports, expands one-dimensional arrays, and preserves the
symbolic-size local variable convention required by Bartiq. The original
refusal of multidimensional ports is retained. Type labels and resource
projections come from the existing model owner. Named selection can unfold
available decompositions, using stable occurrence IDs for child names and
relative endpoints. Source-compatible logical call graphs preserve symbolic
repetition counts and zero expressions.

Two native Program fixtures reproduce the source two-CNOT topology,
emitted operands, output bindings and full complex amplitudes. Both emit
two CX gates, and both have a source Clifford count of two. The crossed
fixture returns the two physical wires in reversed order. These fixtures
have no private work or measurement feedback. Export emits zero instructions
and leaves the existing append path authoritative.

The source decomposed call-graph path clears parent ports before connections,
which causes QREF assignment validation to raise for the connected CNOT
fixtures. Native construction clears both fields together. It retains
the two declared child occurrences and adds the counted repetition.

The native notebook uses weights 0 through 23 with precision (10^{-5}).
The full alias preparation topology, port widths, five routines and
resources match the source QREF model. Recompiling child costs gives the
same historical total. The pinned and alternative uniform preparation
formulas remain visible. Substituting the qualified child parameter at
24 gives 313 T, and the supplied custom function at 101 evaluates to
16 T. The original notebook's unqualified `N` leaves the child parameter
symbolic. Both notebooks execute in separate kernels and render two SVGs.

| Alias preparation measurement | Result |
| --- | --- |
| Pinned historical model | T307, Clifford1770, rotations2 |
| Native default model | T307, Clifford1582, rotations2 |
| Native appended primitive stream | H244, X113, T156, Tdg151, CX1145, S68, measure68, conditional CZ68, reset68, Rz2 |

The existing alias-preparation regressions cover small-state complex
amplitudes and work cleanup. The notebook's 24-input case compares the
complete model and counts emitted instructions without allocating a dense
state. Generic foreign component reflection and native replay of the
larger chemistry and factoring fixtures remain open under Q-0794 and
Q-0795 in that historical batch. The complete input conversion and test replay
are addressed below; the source connected-callgraph failure remains distinct.

## Complete source-boundary interface

Ecosystem owns reflection over optional live SDK components and immediately
converts signatures, instance identities, directional endpoints and declared
historical T/Clifford/rotation values into existing native records. Compiler
continues to own QREF construction. Source call-graph multiplicities and
resource expressions are captured metadata, not native numerical execution or
a claim that every source constructor is executable. No source object or
callback remains in the resulting schema. Existing native metadata consumers
continue to use the same exporters.

Registers preserve names, quantum/classical storage widths, shapes and sides.
QREF has no register-encoding field; these width-only opaque declarations do
not assert executable dtype admission. They deliberately avoid the unrelated
source wire codec, which fails on valid classical CBit metadata in the RSA
fixture. Array elements retain their
declared order in graph endpoints; display ports follow the source's per-register
lexicographic name order, including indices ten and above. Explicit native
occurrence lists preserve disconnected zero-port scalar calls which cannot be
discovered from edges. Their presence also records an available empty
decomposition without changing the legacy serialized library schema. This
export-only path owns no quantum work, measurements or physical gates.

## Complete implementation and consumer disposition

| Frozen source definitions | Native responsibility |
|---|---|
| `_bloq_type`, `_cirq_gate_bloq_type`, `_bloq_instance_name` | Ecosystem metadata labels, exact source flag suffixes and original numeric instance IDs |
| `_is_symbol_or_int`, `_extract_common_bloq_attributes` | Compiler `_attributes`, preserving scalar resource values and local symbolic port sizes |
| `bloq_to_qref`, `bloq_to_routine`, `_bloq_to_routine`, `_default_leaf` alias | Ecosystem source dispatch followed by existing native component exporter |
| `_routine_with_decomposition`, `_composite_bloq_to_routine`, `_bloq_instance_to_routine` | Owned component connections plus explicit complete occurrence lists; selective class matching happens before source types leave Ecosystem |
| `_names_and_dir_from_register`, `_expand_name_if_needed`, `_ports_from_register` | `_qref_ports.py`, including concrete long arrays, empty arrays and unresolved shapes |
| `_opposite`, `_relative_port_name`, `_import_connection` | Existing native directional component endpoints; quantum and classical widths are retained |
| `_ensure_primitive_type`, `_import_resources`, `_extract_input_params`, `_extract_symbols_from_t_complexity`, `_extract_symbols_from_port_sizes` | Ecosystem declared-value projection and Compiler resource/parameter projection; no source cost callback survives capture |
| `_wrap_in_repetition`, `_call_graph_to_routine_map`, `bloq_to_routine_from_callgraph` | Existing native symbolic call graph, collision naming and repeated routines, supplied only integer native identities |

All 25 original function definitions and the final leaf alias have an explicit
owner. No constructor algorithm is inferred from metadata. The source default,
selective and call-graph paths are compared independently against the locked
exporter, not against copied expected summaries. Seven exact source factories
cover the 390-call out-of-place adder graph, both packed and unpacked LCU,
double-factorization one-body and full block encoding, symbolic QROM and RSA
phase estimation. Complete serialized JSON agrees, including resources,
symbolic repetitions, every child, port and connection. Symbolic QROM/RSA
physical decomposition refusal is retained as an explicit separate outcome.

The unchanged original test file passes all 28 cases. Replacing only its
`bloq_to_qref` consumer binding by the native input converter passes the same
28 cases in each actual Cirq profile; the frozen source implementation and
assertions are untouched. Existing native CNOT full-complex operator/output
binding checks and the complete alias/Bartiq model consumer are reused by the
affected regression. A captured double-factorization schema survives full JSON
round trip and topology verification while all source SDK imports are blocked.
The caller's `circuit.append(op)` execution path remains authoritative.

This completes Q-0795's original-test and larger-instance replay responsibility.
Q-0794 remains open under the existing source-failure acceptance rule: its
connected CompositeBloq call-graph exporter clears ports before connections,
raising QREF assignment validation. The existing native exporter clears both
together and yields a valid schema. That reproduced difference is neither
silently patched in the reference nor covered by the approved three-parent
defect exception. Captured historical costs are distinguished from independent
native algorithm price models and from actual emitted primitive inventories.

When a component is unfolded, its exported resources come from the actual
decomposed body, retaining the original instance label; an analytic parent
override must not replace that body price. A typed SDK fixture independently
demonstrates the original 17-Clifford parent view versus its two-X body view.
Named constants such as pi are not misclassified as free port parameters.
The pinned QREF schema owns scalar coercion at the Ecosystem boundary, including
the four NumPy-derived integer values that the original RSA JSON emits as floats.

Final local acceptance passes 119 affected cases in each actual Cirq 1.7/1.6
profile without skips, plus 145 local integration cases. The unchanged original
suite passes 28 cases; its native consumer replacement passes 28 in each profile.
Strict typing covers four Python files using the frozen SDK's real definitions,
without ignoring an untyped base class or supplying a fake SDK contract.
Black/Ruff, Compiler/Ecosystem/verification scopes, English/links, architecture,
IR, API and diff checks pass. Dependency policy retains 724 repository findings
and none on the touched paths. Initial environment, fixture, scalar-type and
strict-check failures are recorded separately from the final accepted receipts.
