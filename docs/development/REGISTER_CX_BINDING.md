# Native CX binding

K-0199 through K-0202 cover the four locked CX files.

| File | Frozen Git blob |
| --- | --- |
| `cx.cc` | `b83879b628ef730c0ca6c499868d0eb467e7d2a4` |
| `cx.pybind.cc` | `de54a2df66f20ffb0e11ada82aa46c9a91c37784` |
| `cx.pybind.h` | `f2b26c8264fd4748e8591549a5d7e488689dd3f8` |
| `cx_test.py` | `f7b50bbfb7a79851faf604cb8b21ec0e50dd7041` |

The native width resolver handles the Python binding's scalar and explicit
sequence inputs. Existing ordered CX emission covers the homogeneous Z
control families and heterogeneous fallback. Quantum targets receive CX,
guarded X or literal X. X-basis outputs produce phase kickback through Z
or NEG and retain the applicable classical guards. Classical targets apply
old-read XOR. Boolean targets, quantum controls on classical targets, and
nonempty X-basis control columns are refused. Width-zero columns have no
operand pairs to admit.

The 16 original scalar types and 81 original broadcast cases match the
effective source stream by count and ordered operand digest. Scalar scopes
are compared by effective guards. The original classical destination
sequence matches its six emitted bit operations. A classical self-alias
emits guarded self-inversion, clears the slot, and preserves the
full quantum amplitudes for both starting values. Duplicate quantum wires
are rejected when the source finishes its circuit. Native admission
rejects them while forming the instruction, before append.

For in-place gates the caller retains all input and output ports. There
are no work wires or measurement results. Mixed controls can select
gates or exact scalar phase, whose native IR carrier is an arbitrary
existing wire. KMX retains NEG separately from its native phase
realization. The generator declares no analytical resource model.
The initial broadcast resolver acceptance separately exercises complete
complex amplitude permutations for sliced quantum arrays.

Run `python -m pytest -q tests/unit/test_kickmix_cx_binding_file.py`.
