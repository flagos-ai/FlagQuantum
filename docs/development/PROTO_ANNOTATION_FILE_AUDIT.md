# Legacy resource annotation source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen files are verified by
their Git blob hashes.

| Source file | Blob | Disposition |
| --- | --- | --- |
| `protos/__init__.py` | `ce76a8d10a5909daeb7b1113c8cc7f26cc31b410` | License-only initializer |
| `protos/annotations.proto` | `3fd709194fe0340b819eb0705883e30236643952` | Standalone proto3 fields and wire tags in the Compiler adapter |
| `protos/annotations_pb2.py` | `ac809603130c9ee4ca3a99a2cb034caff8bf0c16` | Reference generated descriptor verified against the owned private descriptor |
| `protos/annotations_pb2.pyi` | `a8dbed196240727f6f59db30df282fd9f7e1389f` | Fields and constructor checked against typed native byte and cost contracts |
| `serialization/annotations.py` | `a4ebff629a3e10e60a2025a3b91e1bd18c09bf47` | `serialize_resource_annotation` and `deserialize_resource_annotation` |
| `serialization/annotations_test.py` | `bca834902649f4fe1bb871affb422a6eb92f0b12` | Original roundtrip run unchanged and compared to the native byte roundtrip |

`flagquantum.compiler.formats.resource_annotations` maps the three concrete
`GateCost` fields to signed 64-bit values in an implicit-presence proto3
message. Tags 1, 2, and 3 carry Cliffords, rotations, and T gates,
respectively. Default zeros produce empty bytes. The adapter uses the
optional protobuf runtime on demand, with a private descriptor pool.
The descriptor's native message name belongs to FlagQuantum. The
standalone wire bytes remain interoperable.

The legacy budget projection belongs to `gate_budget.py`. An annotation
accepts counts that have already been reduced to the three legacy fields.
The serializer reports an error if factored fields are still present.
Deserialization accepts unknown fields through protobuf and projects the
three recognized values into the model. Malformed bytes and out-of-range
values report `SerializationError`.

Verification compares exact bytes for the original fixture, empty defaults,
nonzero counts and signed endpoints. The modeled projection for T,
Toffoli, controlled swaps, And, measurements and rotations matches the
original GateCounts projection before serialization. The adapter only
reads or writes annotation data. Physical execution and resource inference
belong to the existing owners. Whole-program legacy artifacts remain in
their respective serialization batches.
