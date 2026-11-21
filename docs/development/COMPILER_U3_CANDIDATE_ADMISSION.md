# Target admission for U3 decomposition candidates

The equivalence translator considers an absolute-phase-preserving U3 word
alongside its Euler and registered equivalence candidates. For RZ, SX and SXdg,
the U3 word includes a scalar correction because CircuitIR has no global-phase
field. Previously the translator constructed the leading U3 and both correction
instructions before checking whether the target could execute the first U3.
A target without U3 therefore paid for three U3 instruction constructions while
rejecting that candidate at its first leaf.

The existing private U3 synthesizer now accepts the translator's target
predicate. It checks the leading U3 before constructing the scalar correction.
If that leaf is rejected, it returns no U3 candidate. If it is accepted, every
correction leaf is still checked with its concrete parameters and wires. The
remaining Euler and equivalence candidates retain their original order, ranking,
parameter objects, metadata and failure handling. No cache, new dependency,
public argument, target assumption or serialized field is introduced.

Target-independent one-qubit optimization calls the same synthesizer without a
predicate and retains the complete absolute-phase-preserving word. This change
admits no additional target or capability and makes no performance ranking,
whole-compiler speedup or scalability claim. In particular, it does not establish
that the previously observed direct-CZ translation overhead has been eliminated.

## Existing user workflow

```python
import flagquantum as fq
from flagquantum.compiler.basis_conversion import convert_basis

source = fq.Circuit(1).rz(0, 0.37)
phase_target = convert_basis(source, gates=("phase", "rx", "cz"))
assert tuple(gate.name for gate in phase_target.program.instructions) == ("phase",)

u3_target = convert_basis(source, gates=("u3",))
assert len(u3_target.program.instructions) == 5
assert all(gate.name == "u3" for gate in u3_target.program.instructions)
```

The phase-target form retains the established conversion equivalence up to global
phase. The U3-only form retains the absolute scalar phase through its correction
word. Trainable and symbolic angles remain attached to both the primary rotation
and the correction; nothing is detached or converted to a host scalar.

## Validation

`tests/unit/test_compilation_u3_candidate_admission.py` checks:

- Actual conversion of RZ, SX and SXdg to a non-U3 target constructs one rejected
  U3 instruction instead of three, while independently preserving its operator
  up to the existing global-phase convention and retaining metadata.
- Rejecting the primary U3, the X correction or the phase correction by concrete
  parameter constraints refuses the complete U3-only translation.
- Accepting the U3-only target preserves the absolute complex operator and its
  trainable gradient against an independent gate-matrix product.
- Binding a shared symbolic expression updates the primary and correction gates
  together across zero, positive and negative bindings.

The constructor-count assertions fail on parent
`8004c24560d19fc5fcc21a5668d8dbfcfc5fe6c1`, which constructs three U3 objects
for each rejected candidate. The other five behavioral cases pass there. This
separates reduced construction work from an unsupported change in semantics.

```bash
python -m pytest -q tests/unit/test_compilation_u3_candidate_admission.py \
  tests/unit/test_compilation_basis_translation.py \
  tests/unit/test_compilation_one_qubit_synthesis.py \
  tests/unit/test_compilation_one_qubit_optimization.py \
  tests/unit/test_compilation_basis_conversion.py \
  tests/hybrid_compiler/test_native_gate_legalization.py
```

Existing independent native-CZ application, routing-evidence, growth-budget,
fixed-pulse and unsupported-basis tests remain required. This is local
`single_device_fast_path` development verification; formal SDKBench qualification
remains separate and its participant-neutral protocol is unchanged.

## Mature implementation review

The review inspected pinned primary implementations before choosing the narrow
FlagQuantum-owned change:

- [Qiskit BasisTranslator](https://github.com/Qiskit/qiskit/blob/03c361e882a85150928ff96e6e3e19dac9cbdf46/qiskit/transpiler/passes/basis/basis_translator.py)
  searches from the requested target basis and accounts for qubit-local target
  operations. Its basis-set search is broader than this per-candidate change.
- [Cirq target-gateset transformation](https://github.com/quantumlib/Cirq/blob/e713975b7148e1511a7f99a1c166e74d176cdce5/cirq-core/cirq/transformers/optimize_for_target_gateset.py)
  passes gateset validation into recursive decomposition. This reinforces
  checking concrete target admissibility during decomposition rather than
  assuming that a named decomposition is executable.

Both upstream source files carry Apache-2.0 notices. This change reuses the
existing FlagQuantum synthesizer and target predicate; it copies no upstream
implementation and adds no upstream runtime dependency. It is not a port of
Qiskit's Dijkstra search or Cirq's decomposition protocol.
