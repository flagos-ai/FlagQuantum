# Legacy register declaration wire format source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Source file | Frozen Git blob | Native disposition |
| --- | --- | --- |
| `protos/registers.proto` | `2881112dcbf85f0e30e2a1980f64a9eb45fd7101` | Name, encoding, repeated shape, and nested direction use tags 1–4 |
| `protos/registers_pb2.py` | `b37ff02eee02e07ef42274ead0e3a5b0a42ebd42` | Private descriptors preserve field types, repetition, and enum numbers |
| `protos/registers_pb2.pyi` | `e51a9bf0969b693e6c9495dde8c7a4c7df10e03d` | Constructors accept the same field keywords |
| `serialization/registers.py` | `8f449a4877476b781d5bc081bf79fab5eb3a9ced` | The Compiler codec covers both singular and collection encoders and decoders, including direction conversion |
| `serialization/registers_test.py` | `9517380dea8da66086aa757d2f695c10f1038fd4` | All four original cases run unchanged and their declarations also match native wire bytes |

A declaration holds a name, an existing register encoding, an ordered
integer or symbolic shape, and a direction. The supported register types
and symbolic expressions follow the shared encoding and integer argument
codecs. Direction numbers are LEFT 1, RIGHT 2, and THRU 3. The reference
encoder maps an unsupported direction to UNKNOWN 0. The native encoder
rejects the malformed declaration immediately. Both decoders reject
unknown directions.

A collection holds one repeated declaration field and preserves order,
including the order of two declarations with the same name on distinct
sides. An empty collection uses the empty message. Native `Signature`
checks whether the declarations form a valid signature.

A decoded 2-by-2 control register is consumed by the existing predicate
append operation. The forward and erase streams match the original gate
order, operands, output wiring, and measurement feedback. The coherent
state checks all 16 input amplitudes and the three predicate prefix bits.
Each of the eight erasure histories restores the input state with the
expected branch amplitude and clears all three work wires.

| Stage | Observed physical instructions | Pinned model |
| --- | --- | --- |
| Compute | CX18, H6, T6, Tdg6, X4, S3 | AND3, Clifford4, logical width 7 |
| Erase | X4, H3, measure3, conditional CZ3, reset3 | Clifford7, measurement3 |

The wire codec itself emits zero instructions. The existing predicate
owner supplies the gate stream and resource declarations.
