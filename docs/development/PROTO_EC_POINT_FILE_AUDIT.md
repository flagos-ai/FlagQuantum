# Elliptic-curve point wire format source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The complete source family is
verified by Git blob hash.

| Source file | Blob | Native disposition |
| --- | --- | --- |
| `protos/ec_point.proto` | `c57fa1b53dec69f9617f13e0670103ba76eb14a9` | Four nested integer-or-expression fields with the original tags and optional coefficient presence |
| `protos/ec_point_pb2.py` | `f28f2c1105c8d310fc49e3252794dfcc29187b4a` | Frozen generated descriptor checked against the private native descriptor |
| `protos/ec_point_pb2.pyi` | `ac82158cd6d0d49842a7f49655fb88d98cdb5eac` | Original field types and keyword order compared with the native byte and point-factory contracts |
| `serialization/ec_point.py` | `fb690758868bed239c614610e2d3c73bb582983a` | Compiler point adapter reuses the shared nested oneof codec |
| `serialization/ec_point_test.py` | `14dc066a1789f9881ffef652c2f6b266dd232f31` | Three original cases run unchanged and their bytes are compared with the native encoding |

The adapter writes x, y, mod and curve_a, including an explicitly selected
zero arm. It reads the values through the existing integer and symbolic
decoder. Callers pass their point constructor, and the concrete elliptic
curve point remains owned by Algorithms. Symbolic scalar multiplication
uses the two admitted opaque coordinate function names. Unresolved wider
symbolic syntax is tracked by Q-0864.

A missing curve_a is valid according to the original schema. The original
decoder raises SyntaxError by passing the unselected nested field to the
symbolic parser. The native adapter uses the point owner's zero default
for the absent field. A present yet empty coefficient and missing
coordinates or modulus raise field-specific serialization errors. The
frozen failure and the explicit-zero wire bytes are part of the regression.

Native serialization emits no physical instructions and preserves the
point declaration rather than inferring resources. A decoded point feeds
the same windowed QROM tables and ordered append instructions, including
measurement feedback, as the original native point. The elliptic-curve
regressions independently check complex amplitudes, work cleanup, and
source cost declarations for the execution path.
