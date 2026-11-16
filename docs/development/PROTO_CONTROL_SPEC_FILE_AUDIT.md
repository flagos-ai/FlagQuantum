# Legacy control specification wire format source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Git blob hashes identify the frozen
source family.

| Source file | Blob | Native disposition |
| --- | --- | --- |
| `protos/ctrl_spec.proto` | `38415ce602abd14ee5032f1c3047101f13116690` | Two ordered repeated messages, quantum types at tag 1 and NumPy payloads at tag 2 |
| `protos/ctrl_spec_pb2.py` | `6e55523682744103e3710f850fb30fac727ae59c` | Private native descriptor with the original repeated fields and nested contracts |
| `protos/ctrl_spec_pb2.pyi` | `778e150a702ff44936a6e7cb897963b2ef2eaa89` | Frozen constructor fields and repeated message containers checked |
| `serialization/ctrl_spec.py` | `30895b51e932e2efa2571dac19373a210fb3a5a3` | Compiler codec reuses the register descriptor and NumPy .npy owners |
| `serialization/ctrl_spec_test.py` | `c7415d0a2b7a4648af1c282439858b13c630caf5` | Five original tests run unchanged, and all five values match native bytes |

The format keeps a tuple of paired native register types and NumPy control
arrays. Rank-zero controls, array shapes, dtypes and register order survive
the roundtrip. The two repeated lists must have equal lengths. As in the
source, symbolic Shaped values have no wire representation.

Concrete clauses reach the existing integer predicate owner. Array elements
are visited in C order and each word is expanded in big-endian bit order.
Signed, ones' complement, and raw fixed-point integer activation use their
existing encodings. A symbolic-width declaration can stay in the wire
format. Binding a physical predicate requires a concrete width.

Three decoded four-control fixtures feed the normal predicate append
operation. They match the pinned forward and erase streams, operands
and measurement feedback. All full complex input amplitudes match the
expected computed prefix state. All eight erase branches per fixture
restore the control state and clear the three work wires.

| Stage | Observed physical instructions | Pinned model |
| --- | --- | --- |
| Compute | CX18, H6, T6, Tdg6, X4, S3 | AND3, Clifford4, logical width 7 |
| Erase | X4, H3, measure3, conditional CZ3, reset3 | Clifford7, measurement3 |

The wire codec itself emits zero instructions. The separate predicate
owner supplies the gate stream and resource declarations.
