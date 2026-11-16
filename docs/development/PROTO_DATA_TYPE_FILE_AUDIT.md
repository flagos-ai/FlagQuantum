# Legacy register-type wire format source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Git blob hashes identify the complete
source family.

| Source file | Blob | Native disposition |
| --- | --- | --- |
| `protos/data_types.proto` | `2848ffb0a07e0e2b161c332c884202c54a4ae846` | Eight quantum type alternatives with the original tags and optional fields |
| `protos/data_types_pb2.py` | `e7e669456dc1abe8727d84553a59bfc04b22f946` | Frozen generated descriptors checked against private native descriptors |
| `protos/data_types_pb2.pyi` | `ec9125233e1f257e1efa307b03e055aae3ad9e42` | Original constructor fields and presence matched by typed byte and RegisterType contracts |
| `serialization/data_types.py` | `9f7aebe00f0d69d3b447e08309627f8978344b17` | Compiler encoder and decoder reuse the existing Core factories and nested integer-or-expression codec |
| `serialization/data_types_test.py` | `ff88e4e11b781bbcedf773535c9c66f750c843d3` | The 13 unchanged tests and all their 25 concrete or symbolic values |

The eight wire alternatives are bit, opaque, signed, ones' complement,
unsigned, bounded unsigned, fixed-point and Montgomery. Symbolic bit
widths and bounds use the existing admitted text grammar. Decoding uses
the same Core constructors as native callers. A missing fixed-point
signed flag reads False. Encoding its False value selects that optional
field. Required numeric subfields with an absent oneof report a
serialization error.

The original Montgomery message stores the width alone. The frozen
witness with modulus seven roundtrips to modulus None. The native decoder
accepts the original bytes and reconstructs a width-only declaration.
Encoding a native declaration with a modulus raises a serialization error
to prevent dropping the modulus. Classical storage and the other
unrepresentable encodings have explicit refusal paths. The wire schema
and its limitations are documented independently from numeric support.

The QROM check feeds wire-roundtripped native declarations into the same
production signature and placement path. The ordered operands and
primitive stream agree with the original native declarations. The
serializer carries metadata and emits zero instructions. The existing
type and encoding owners govern numerical values and physical admission.
