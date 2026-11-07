# Out-of-place addition: shared resources

The register compiler owns the existing MSB-first emitter in
`out_of_place_addition.py`, the native call recipe in `arithmetic_declarations.py`,
and the new shared model in `out_of_place_add_resource_model.py`.
Algorithms compose the model; they do not repeat its gate formulas.

Inputs a and b each have n bits. The sum c is a logical output of width n+1
when `include_msb=True`, otherwise n. It is not an allocated work register.
The measured adjoint assumes the correct logical sum and erases it. Existing
source gate, full-complex-state and measurement-branch tests remain authoritative
for execution; the model neither emits nor simulates an operation.

`source_out_of_place_add_resource_formula` accepts native or symbolic widths,
`include_msb`, `adjoint`, and the explicit logical QEC profile. It queries shared
native declarations. `source_out_of_place_add_cost` accepts Python integer widths
and returns the same estimate as a composable ComponentCost. Grid PairPotential
now consumes it for the three forward sums and three measured inverses.

The unchanged reference is Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`,
`qualtran/bloqs/arithmetic/addition.py`. Its direct declaration is
`And: n+int(include_msb)-1` and
`CNOT: 5*(n-1)+2+3*int(include_msb)`; the adjoint adjoints the And.
The implementation queries those recipes rather than repeating these equations
in the model or algorithm consumer. Concrete peak width is 3n+int(include_msb).
Symbolic queries retain the original Max(3, 2n, 3n+int(include_msb)) fallback.

Python zero width fails the original QubitCount query and is rejected by the
combined native formula. Sympy Integer(0) remains a distinct declaration query
with peak 3, including the truncated negative gate-count diagnostic. These cases
are not normalized into executable zero-width operations. The source prose's
unqualified 4n-4 T claim does not override its full-width executable loop.

## Model and emitted counts

At eight input bits, the independently tested emitter and the unchanged source
have the same primitive stream. The table keeps that stream separate from the
logical QEC model (default profile):

| Output | Direction | Model And | Model Clifford | IR CX | IR H | IR S | IR T/Tdg | Measurements | Logical qubits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| truncated, 8 bits | forward | 7 | 37 | 79 | 14 | 7 | 28 | 0 | 24 |
| truncated, 8 bits | adjoint | 0 | 44 | 37 | 7 | 0 | 0 | 7 | 24 |
| full, 9 bits | forward | 8 | 40 | 88 | 16 | 8 | 32 | 0 | 25 |
| full, 9 bits | adjoint | 0 | 48 | 40 | 8 | 0 | 0 | 8 | 25 |

Adjoint IR also has one conditional CZ and one reset per measurement. Model
Clifford counts are abstract logical categories; they are not the emitted
primitive Clifford total. Primitive T counts close as four times model And;
measurements close directly. No extra quantum work is required beyond a, b, c.

Run:

```sh
python -m examples.algorithms.arithmetic_resource_graph
python -m pytest -q tests/unit/test_out_of_place_add_resource_model.py tests/unit/test_out_of_place_addition.py tests/unit/test_grid_potential_resource_model.py
```

Reference tests compare direct callee order, polarity/direction, both logical
cost profiles, every resource category and concrete/symbolic width fallback.
Replacement tests swap the declared recipe and the algorithm's model dependency
without changing their consumers. Existing user code keeps `circuit.append(op)`;
this batch changes resource composition and leaves executable gates unchanged.

This batch adds 132 tests and passes 854 concentrated checks without skips,
including the pre-existing complex-state/measurement-history regressions. Changed
production modules and the example pass strict mypy, Black, Ruff and architecture
checks. Source addition and grid-potential blob IDs match the pinned inventory.
