# Factoring Fourier-measurement shim

Q-0453 and Q-0454 cover the license-only cryptography initializer
and `_factoring_shims.py` from Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Both pinned files match
their Git blobs. The source file's only operation is `MeasureQFT`.

The locked signature declares a shaped QBit array named `x` on
the left. Its direct calls are one forward `QFTTextBook` and one
`MeasureZ` per element. The native source declaration preserves
those calls and the input signature at concrete and symbolic widths.
The concrete gate budgets match the source, and the textbook child
peaks match its original qubit-count query. The symbolic resource
formula preserves the source's grouped phase-gradient fallback.

For positive widths the original builder leaves every returned CBit
unused at finalization. Its concrete composite construction and
top-level qubit-count query fail. At width zero the source builder
also fails because the empty textbook QFT has no qubit port. The
native empty binding is an identity. The native positive-width
operation keeps the measurements explicit and records their outcomes
in caller-owned classical slots. It applies the existing textbook
forward QFT with reversal, followed by Z readout in register order.
The measured wires remain projected onto their observed values;
this operation has no work wires. The separate Kickmix demolition
emitter owns its own feedback and clearing semantics.

| Scope at width four | Evidence |
| --- | --- |
| Native emitted primitives | Four H, six controlled phases, two swaps and four measurements |
| Locked declared gate budget | Two T, six And, fifteen Clifford, one rotation and ten measurements |
| Native logical model width | Peak five from the textbook child's declaration |

The native tests cover all projected complex branches at widths one
through four with an untouched spectator and permuted wire and slot
bindings. Their Fourier oracle is the analytic DFT matrix. The
textbook QFT regression independently matches the ordered
source-derived primitive gates. A trajectory test records a
deterministic word in noncontiguous result slots.

```python
from flagquantum.algorithms.primitives.measure_qft import MeasureQFT
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2).append(
    MeasureQFT(2).make_on(x=(1, 0), result_bits=(5, 2))
)
assert [gate.metadata["classical_bit"] for gate in circuit.to_ir().instructions
        if gate.name == "measure"] == [5, 2]
```
