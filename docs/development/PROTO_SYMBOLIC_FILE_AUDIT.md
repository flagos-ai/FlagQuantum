# Structured symbolic wire format source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The five source files are verified
by their Git blob hashes.

| Source file | Blob | Native disposition |
| --- | --- | --- |
| `protos/sympy.proto` | `0afe51aab0f228faea86be740396e69f20b9c3c1` | Four recursive messages, two enums, two oneofs and repeated operands |
| `protos/sympy_pb2.py` | `c29c7cbcad61b6d1fe92074e3d0fe17864e4ec97` | Frozen generated descriptor compared with the private native bootstrap |
| `protos/sympy_pb2.pyi` | `868aae3c3f43224a4439b7b486bec8d6227e1528` | Original message constructors and field names compared with the descriptor |
| `serialization/sympy_to_proto.py` | `eefe901f0428c8c34e0f856143585cf2af0b1710` | Compiler's typed symbolic byte adapter |
| `serialization/sympy_to_proto_test.py` | `d1853abe918ea0cd8070a467337c0ed284deffef` | All 21 original tests run unchanged, with native exact byte comparisons |

The decoder uses explicit SymPy constructors for all 13 source function
tags and five constants. It preserves recursive operand order, exact
rational arithmetic, signed int32 fields and float32 precision. Subterms
can hold another term or an atom. A function-free term supplies exactly one
operand. Unknown tags, absent oneofs, invalid arity and oversized numeric
powers raise a serialization error. The structured adapter shares the
optional protobuf runtime and expression power limit with the earlier
standalone wire adapters.

The source expands a single symbol name containing spaces into a tuple
during decoding. The native decoder reconstructs the literal scalar
`Symbol(name)`. The name-only wire format omits symbol assumptions.
An int32 overflow, unsupported expressions such as Piecewise, and a
malformed rational denominator have explicit refusal paths. The
separate string grammar remains tracked by Q-0864.

All original supported fixtures match the frozen wire bytes and decoded
value types, including the source float precision result. The complete
native descriptor matches all original fields, enum values and presence
semantics after normalizing the file and package names. This serialization
path carries symbolic values and emits zero circuit instructions.
