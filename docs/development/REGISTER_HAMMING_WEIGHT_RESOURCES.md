# Hamming-weight declarations and resources

The existing `hamming_weight.py` still owns the concrete three-to-two count tree
and measured erasure. Its new `hamming_weight_resource_widths` owns the source
logical junk/output declarations; it never allocates or executes symbolic gates.
`hamming_weight_resource_model.py` owns source aggregate queries and explicit
concrete allocation models. `hamming_weight_phasing.py` consumes the shared
compute/erase models instead of duplicating their And/CNOT equations.

The pinned unmodified source is Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`,
`qualtran/bloqs/arithmetic/hamming_weight.py`. The phasing consumer comes from
`qualtran/bloqs/rotations/hamming_weight_phasing.py`. Copyright/source attribution
is retained. No upstream class or source is imported by production code.

## Interfaces and declared graph

`ArithmeticCall("hamming_weight", (width,), adjoint=...)` reuses the existing
native resource identity, graph and QEC query machinery. The logical source word
is THRU; junk and output are RIGHT for compute and LEFT for erase. Existing MSB
binding, little-endian tree traversal and logical junk ownership are unchanged.
Forward may modify the source; erase requires the entire correct computed state.

The exact source declarations are:

- bit_count = Python width.bit_count() for Python integers; otherwise 1.
- junk = width - bit_count; output = existing bit_length(width).
- compute calls And `junk` times and CNOT `5*junk + bit_count` times.
- the resource adjoint adjoints the And leaf in place; executable reverse order
  remains the concrete emitter's separate responsibility.

All seven resource categories use the generic query and original cost profile.
`source_hamming_weight_cost` returns the concrete declared local storage for use
by phasing. `source_hamming_weight_resource_formula` additionally matches the
original successful QubitCount query, including its fallback maximum over
signature, input width and the three-qubit And leaf for symbolic inputs.

## Source boundaries retained

| Width object | Compute And / Clifford | Source query peak |
|---|---:|---:|
| Python 3 | 1 / 7 | 6 |
| Sympy Integer(3) | 2 / 11 | 7 |
| Unknown integer n | n-1 / 5n-4 | Max(3, n, 2n+ceiling(log2(n+1))-1) |
| Python 0 or 1 | Raw declarations remain inspectable | Source QubitCount fails; aggregate formula raises |
| Sympy Integer(0) | -1 / -4 | 3, diagnostic model data |

Sympy zero's erase model returns Clifford=-5 and measurement=-1. These original
negative declarations are retained for source fidelity and are not valid
physical budgets. The concrete emitter/allocation model rejects zero width.
Python width one's native two-wire CX implementation and explicit allocation
model remain usable; they do not constitute a successful source QubitCount or
empty-junk adjoint decomposition. That upstream failure remains open.

For eight input bits, compute+erase uses 19 live qubits: eight source, seven
logical junk and four logical count. Primitive scans give 28 T/TDG, 114 CX and
seven measurements; junk/count clear after correct erasure. The independent
logical model gives seven And, 79 Clifford and seven measurements across both
directions. Its And conversion gives 28 T-equivalent. Abstract Clifford counts
are separate from expanded primitive CX/H/S/feedback instructions.

## Acceptance and examples

G2-HW02 adds 123 tests: 48 category/peak queries, 20 original direct graphs,
12 original port-width queries, 32 concrete allocation/category comparisons,
four failed Python zero/one source queries, one Python/Sympy divergence witness,
one phasing dependency replacement, four invalid parameters and one diagnostic
literal-zero witness. Both adjoints and both QEC profiles are included. The
replacement supplies all seven gate categories and larger private child peaks,
proving the phasing consumer composes the shared models.

The combined suite passes 1,491 tests without skips, retaining six existing LCU
reference warnings. It verifies existing source gate dependencies, full complex
measurement branches, logical outputs, clean work and phasing/gradient consumers.
Six production/example files pass strict typing; Black, Ruff, scope, architecture
and all three examples pass.
Run the existing append workflows and model example:

```sh
python -m examples.algorithms.register_hamming_weight
python -m examples.algorithms.register_hamming_weight_phasing
python -m examples.algorithms.arithmetic_resource_graph
```

The resource example explicitly retains the original width objects. No symbolic
execution or successful source query on a failing boundary is claimed. Source
parent files remain open in the integration checklist.
