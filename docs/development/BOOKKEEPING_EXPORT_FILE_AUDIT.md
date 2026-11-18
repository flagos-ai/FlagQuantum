# Complete bookkeeping package exports

Q-0308 covers the complete initializer at locked Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`3dd8073aa1f9cb05ff8f812de6c60d6879b49607`. Its module docstring and eleven
imports are accounted for. The source test checks every export in original
order and its identity with the original defining module, including the unusual
Join2/Partition/Split2 import order. No gate algorithm is defined here.

| Original export | Existing native responsibility and consumer |
| --- | --- |
| Allocate | Compiler `component_builder.py` allocation, Core Program metadata and physical binding and existing clean/borrowed pools. `test_controlled_workspace_lifetimes` checks lifecycle events, caller ownership and borrowed-state restoration. |
| Always | Compiler `registers/primitives/always.py` and `always_component.py`. `test_always_source_binding` and `test_always_component` retain the same wrapped child and operation order. |
| ArbitraryClifford | Compiler `_component_trace/resource_generalization.py` owns opaque Clifford signature and cost identity. `test_arbitrary_clifford_resource` checks concrete and symbolic declarations. This is a declaration without a synthesized arbitrary Clifford body. |
| AutoPartition | Compiler `registers/layout/auto_partition.py` and `component_auto_partition.py`. Source binding and component suites preserve partitioned groups, unused fields and child traversal. |
| Cast | Compiler `layout/register_cast.py`, typed Program construction and binder. `test_register_cast`, `test_component_cast` and `test_cast_signatures` retain same-shape retyping, exact operands and direction. |
| Free | Compiler builder ownership consumption plus physical pool return. Lifetime tests preserve clean obligations and original allocation-pool identity. A Free marker does not imply reset. |
| Join | Compiler `component_builder.py` join and `layout/register_bits.py` rebinding. `test_register_split_join` retains ordered bit ownership and zero-width metadata. |
| Join2 | Compiler `registers/layout/partition.py` and Compiler typed rebinding. `test_register_partition` retains inverse two-group partition semantics. |
| Partition | Same partition owner with declared group widths and sides. Component and source binding tests retain concrete tensor axes and logical output aliases. |
| Split2 | Same partition owner with original forward two-group partition semantics. |
| Split | Compiler `component_builder.py` split and `layout/register_bits.py` rebinding. `test_register_split_join` checks per-bit direction, MSB-first order and single consumption. |

## Interface and evidence reuse

Metadata and ownership remain in Core, physical instruction binding in Compiler,
and numerical callbacks/tensor contraction in Simulation. These exports require
no new top-level FlagQuantum names. Existing Program-consuming operations retain
`circuit.append(op)` and use the accepted binding pipeline. Explicit consumer
suites in this batch cover actual source capture, retyping, lifetime sequences,
complex states, clean scratch, borrowed joint states and applicable measured
cleanup. Accepted evidence is reused with its existing domain boundaries.

Bookkeeping rebindings emit zero additional primitives. Allocation changes live
capacity, while dirty borrowing must restore the original joint state. Free
preserves its clean/dirty precondition and pool contract. Source QubitCount is
queried independently of caller-reserved capacity and measured child primitives.
Always preserves the child's model, and opaque ArbitraryClifford retains its
own declared category. The initializer itself has zero work and branches.

Q-0308 closes only the complete export surface. Generic descriptors, full dtype
and foreign-hook admission, symbolic physical binding and source drawing gaps
remain under their existing implementation parents. Original undefined or
unsupported paths are not inferred from successful concrete ownership tests.
The original source files remain unchanged.

The shared batch passed 792 concentrated native/reference cases without skips
or warnings. All four changed Python test files passed strict mypy, Black and
Ruff. Compiler scope, architecture, multilevel IR and protected API checks
passed. The source initializer blobs and child reference files were unchanged.
Logs are `ftqc-reference-env/g2-package-exports-*.log`.
