# Coherent unsigned minimum

The user writes a conditional value, `a if a < b else b`. The compiler owns the
comparison, reversible selection, scratch storage and cleanup; the program
contains no gate definitions, physical wire numbers or manual bitstring decoding.
It preserves the original constant input four and a uniform three-bit input.

```sh
python -m examples.synthesis.functions.arithmetic.extremum.minimum_2vars_example.main
python -m examples.synthesis.functions.arithmetic.extremum.minimum_2vars_example.comparisons.circuit
```

The comparison constructs a reversible XOR truth table independently of the
compiler. It checks the complete complex state including clean hidden workspace.
Both native and original programs use 15 qubits, 143 operations and depth 118;
all six workspace bits return to zero. Generic acceptance also covers maximum,
equality ties, nested and constant
predicates, wider branches, complex phases and checked helper composition.

The sibling extremum Notebook also contains `max(3.5,a)`. Its fractional
semantics are not covered here; that complete Notebook remains pending.
