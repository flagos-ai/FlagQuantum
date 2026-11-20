# Legacy argument wire format source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen files are
verified by their Git blob hashes.

| Source file | Blob | Native owner |
| --- | --- | --- |
| `protos/args.proto` | `f967f4a41c47ba005fe82e8018f08a67e43d1303` | Three wire descriptors in `legacy_arguments.py` |
| `protos/args_pb2.py` | `65f741b5a144f58b8914479cd609be3be1cc60b2` | Private runtime descriptors with matching fields and tags |
| `protos/args_pb2.pyi` | `5ea037cbb4d22a74df33f76f14ac1708726e44dc` | Typed native byte and value contracts |
| `serialization/__init__.py` | `a79cdd57a577d220fc8fa47a94186302dfda6aa0` | The existing Compiler formats namespace has the same docstring-only initialization |
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
selected arithmetic functions such as `floor`, `sqrt`, and `log`.
Piecewise values may use comparison, equality, inequality and Boolean
predicates, with nested branches. The arithmetic grammar also admits
`Mod`, `KroneckerDelta`, `binomial`, and `factorial`.
Unknown calls, attribute access and subscripts raise a serialization error
inside branches as well as at the root. Integer, exponent and combinatorial
checks bound eager evaluation. Missing oneof arms, object arrays and archive
containers also report explicit errors.

The source numeric encoder rejects standalone relations, while the generic
source decoder accepts standalone relational text supplied directly in the
wire message. Native numeric decoding rejects Boolean-valued roots. The
source grammar also accepts additional numeric constructors, and the text
format loses symbol assumptions when only the name is printed. These
boundaries remain tracked in Q-0864.

A piecewise dimension roundtrips through register declarations and a
library count map with source bytes intact. At the two selected parameter
values, the decoded shape binds three and four controls through normal
append. The tests compare the forward and erase gates, operands and
feedback. Each of the four and eight respective erasure histories restores
the input complex amplitudes with the expected branch factor and clears the
work bits.

| Controls | Observed instructions | Source QEC model |
| --- | --- | --- |
| 3 | 8 T/Tdg and 2 measurements | 2 AND and 2 measurements, logical width 5 |
| 4 | 12 T/Tdg and 3 measurements | 3 AND and 3 measurements, logical width 7 |

Wire tests cover signed int64 endpoints, exact float32 bytes, structured
and endian-specific dtypes, scalars, empty arrays and Fortran order.
The shared protobuf factory uses the optional runtime on demand and
serves both arguments and standalone resource annotations. The format
layer carries values and emits zero quantum operations. Compiled program
formats and execution admission are owned by their existing contracts.

## Generated count symbols and product grouping — 2026-11-20

Actual ArcTan depth-one libraries contain the source allocator's `_c0` atom.
The shared bounded parser now accepts single-underscore symbol names, including
inside Piecewise predicates. Double-underscore names, attribute access, indexed
values and arbitrary calls retain their refusal. The AST builds symbolic nodes
without resolving names to Python objects.

The actual symbolic ArcSin cost also exposes a structural codec defect. Binary
left-associated multiplication distributed a scalar into an intermediate sum,
changing the original printed expression despite equal mathematics. The parser
now collects factors into one SymPy product. The unchanged source model then
roundtrips with exact wire bytes at depths zero, one and two. The trigonometric
foreign tests protect that real library. Existing integer/exponent/combinatorial
budgets and unsupported-code cases remain active.
