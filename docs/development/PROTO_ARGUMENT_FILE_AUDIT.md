# Legacy argument wire format source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The five frozen files are
verified by their Git blob hashes.

| Source file | Blob | Native owner |
| --- | --- | --- |
| `protos/args.proto` | `f967f4a41c47ba005fe82e8018f08a67e43d1303` | Three wire descriptors in `legacy_arguments.py` |
| `protos/args_pb2.py` | `65f741b5a144f58b8914479cd609be3be1cc60b2` | Private runtime descriptors with matching fields and tags |
| `protos/args_pb2.pyi` | `5ea037cbb4d22a74df33f76f14ac1708726e44dc` | Typed native byte and value contracts |
| `serialization/args.py` | `68c2b3136d36287d07c2814ed520eaecd98d2425` | Six native encode/decode functions and an explicit symbolic text grammar |
| `serialization/args_test.py` | `d099ec8fde0a01f3443218c362556bd4f8a85b15` | Five tests run unchanged, with separate native wire comparisons |

The signed int64 and expression string arms retain their proto3 oneof,
including the distinct presence of explicitly encoded integer zero.
Arrays carry a NumPy .npy byte stream with pickle disabled. The source
complex wire carries two float32 fields, so roundtrips preserve precisely
that precision. The source complex decoder's type hint names
`IntOrSympy` while its implementation reads the actual `Complex` fields.
The native decoder binds the dedicated complex descriptor.

The symbolic decoder builds SymPy nodes from a bounded AST. It admits
scalar numbers, named symbols, selected constants, arithmetic, and
selected arithmetic functions such as `floor`, `sqrt`, and `log`. Arbitrary
Python calls, attribute access, and subscripts raise a serialization
error. The source uses the general SymPy expression parser. Wider
symbolic forms such as piecewise expressions and relational types remain
under Q-0864. Missing oneof arms, oversized integer powers, object arrays,
and archive containers also report explicit errors.

Wire tests cover signed int64 endpoints, exact float32 bytes, structured
and endian-specific dtypes, scalars, empty arrays and Fortran order.
The shared protobuf factory uses the optional runtime on demand and
serves both arguments and standalone resource annotations. The format
layer carries values and emits zero quantum operations. Compiled program
formats and execution admission are owned by their existing contracts.
