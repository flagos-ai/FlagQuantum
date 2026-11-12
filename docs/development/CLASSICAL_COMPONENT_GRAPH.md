# Classical assignment graphs

`export_classical_component_graph_dot` projects computed SSA assignments onto the
existing typed component register graph. It follows all three methods in locked
Qualtran v0.7.0 `drawing/classical_sim_graph.py` at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

## Ownership and interface

| Source method | Native responsibility |
| --- | --- |
| `__init__` | Caller runs existing Simulation `ComponentClassicalState`; Drawer receives the resulting assignment mapping explicitly |
| `cxn_label` | Drawer resolves the connection's producer SSA value and stringifies its assignment |
| `cxn_edge` | Existing register graph geometry with value label, floating label, font size 10, dark-blue text, dot arrowhead and arrow size 0.25 |

The new owner is `flagquantum/drawer/classical_component_graph.py`. The existing
`register_graph.py` shares its renderer through one private edge-attribute
callback; the ordinary width-labeled export retains its signature and output.
Component score, register lifetimes, instance ordinals and lane assignment remain
owned by their existing modules. No numerical execution or external source
objects enter Drawer. No second graph IR, state evaluator or backend registry is
introduced. Source type-only aliases and imports are replaced by native mapping,
ProgramRecord and ValueRef types; no source classes or global side effects are
left unaccounted for in this small source file.

The finite logical component admission is unchanged: positive concrete quantum
words, row-major array elements, MSB-first words and unconditional typed calls.
Caller-owned mapping entries retain separate SSA values for successive outputs
on a THRU lane. Labels use the producer, including root inputs and idle lines;
they never look up the consumer's overwritten result. Output-only and consumed
lifetimes use existing allocation/release geometry. Absent producer values raise
KeyError; stringification errors propagate. DOT string quoting escapes quotes,
newlines and backslashes. Neither the record nor mapping is mutated.

A caller with a completed state uses:

```python
state.simulate()
dot = export_classical_component_graph_dot(record, state.assignments)
```

The computation and drawing are separate entry points. This also permits drawing
explicit diagnostic assignments or replacing the evaluator without changing the
drawer. Circuit construction still uses the same `circuit.append(op)` path.

## Source and native checks

The complete original MultiAnd fixture is retained: four controls, three And
instances and thirteen connections. Both activation patterns `(1,1,1,1)` and
`(1,1,0,1)` cover all sixteen input bit strings. Every connection's register,
producer, consumer and value is compared to the unchanged source drawer. Edge
presentation attributes are compared to the original `cxn_edge` result.

A two-X THRU chain explicitly checks values `0,1,0`; reading a consumer output
would give incorrect labels. Drawing is rerun with numerical execution disabled.
Scalar, vector, matrix and empty idle ports, escaping, absent/incomplete state,
unknown entry, stringification failure and closed state/child/effect lifetimes
are tested. The closed graph is rendered through real Graphviz. The sole original
`classical_sim_graph_test.py` function also runs with its assertions unchanged and
a native boundary replacement, retaining the original MultiAnd input `[1,1,0,1]`.
Only DOT quoting is normalized for that source test's numeric-label regex.

Q-0736 remains open for the general foreign Bloq/PrettyGraphDrawer admission and
constructor interface. The original drawing notebook is a separate inventory
asset and is not marked complete here. Q-0737's single concrete test is audited at
its own file scope; it does not certify the still-open general drawing parent.

## Physical behavior and accounting

This projection adds no primitive gates, workspace, measurements or feedback.
Binding before and after export produces exactly equal physical programs.
Forward MultiAnd's junk/target are live logical outputs, not work claimed to be
erased by drawing. Existing predicate compute/erase and measured Add tests cover
complex amplitudes, all small measurement branches, work zero and ordinary append.
Physical instruction counts and unchanged source analytical QEC costs are recorded
separately in the batch resource evidence; graph edges are not quantum gates.

For the rendered four-control positive predicate, actual native instructions are
H6, T6, Tdg6, CX18 and S3, with no measurement or feedback in the forward graph.
The unchanged source model reports AND 3. Both accounts retain their own layer;
no conversion of AND 3 into a claimed measured primitive total is made. Rendering
with Graphviz 16.1.0 produces thirteen value-labeled edges; the output boundaries
use native lifetime names such as `call0:target` instead of source aliases `junk`.
This is inherited component-score naming, not a promise of identical pixels.

## Local acceptance

Implementation `312df299` adds eleven cases. The final concentrated regression
passes 2,919 tests in 67.36 seconds (35 retained warnings), plus 84 CPU/TN tests in
5.29 seconds (two warnings); no cases skip. Strict typing covers three Python
files. Black, Ruff, Compiler scope, architecture, multi-level IR and protected API
checks pass. Both drawing owners have 100% statement/branch coverage (71 statements,
20 branches). Six frozen source blobs match. The original source-only test passes
in 0.39 seconds, and the native-boundary execution retains its sole function's
assertions. The measurement-uncomputation example runs successfully.

Evidence uses `g2-classical-component-graph-` in the reference workspace. The
initial missing-Graphviz regression log is retained; Graphviz 16.1.0 was restored,
then the affected regression was rerun and the real SVG/PNG inspected. The old
default-suite timeout is not rerun. Q-0737 is accepted at its one-test file scope;
Q-0736 and the graphviz notebook asset remain independently open.
