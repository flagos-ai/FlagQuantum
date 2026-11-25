# Hidden bitstring recovery

The native program expresses a linear phase oracle, surrounded by two register
Hadamard transforms. The compiler chooses physical wires and emits the gates.
`build_program(13)` binds the five-bit secret as classical coefficients; it does
not prepare the answer directly. The same program supports every secret in 0..31.

```python
import flagquantum as fq
from examples.synthesis.algorithms.foundational.bernstein_vazirani.main import build_program

program = build_program(13)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x",))))
# {(13,): approximately 1.0}
```

`main.py` composes reusable quantum functions, prints a circuit diagram, executes
locally, draws 1000 seeded samples from named exact probabilities, and recovers
the secret from the most frequent outcome. This covers all computational steps
of the reviewed Notebook. Local text drawing replaces the cloud circuit viewer;
standard-library sampling from exact probabilities replaces the remote sampling
session. It does not reproduce a hardware job or claim a hardware measurement.

The phase oracle is equivalent to preparing a minus-state work qubit, applying
the original CX predicate, and unpreparing that qubit. The Circuit comparison
checks both forms, including the work qubit's zero state and absolute phase.
Secret 13 is checked against the unchanged original source in test fixtures;
secret 22 is an additional independent case. Acceptance tests check every one of
the 32 secrets, invalid bindings and the complete draw/sample/recover workflow.

For secret 13, the optimized native circuit uses five allocated qubits, three
operations, depth one and no workspace, matching the frozen original resource
row. H/phase/H sequences can be fused into single-qubit operations by the existing
compiler. No new compiler optimization was added in this migration.

The native program specializes the tutorial's linear oracle with classical
coefficients. It does not add arbitrary callable quantum ports or broaden the
native `within` contract to allow arbitrary gate preparations. The two explicit
Hadamard transforms are each other's inverse.

Notebook identity and reviewed cell hashes appear in [provenance](provenance.json).
The shared [third-party license](../../../CLASSIQ_LICENSE.txt) applies.
