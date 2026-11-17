# Register swap binding

K-0208 through K-0210 cover the locked physical SWAP implementation and
Python binding declarations.

| File | Frozen Git blob |
| --- | --- |
| `swap.cc` | `9021c1181802fe771e352e88f5f6dfa51b7d7e92` |
| `swap.pybind.cc` | `bbc9feba7b97e0dfa1d65d427fc4f9ffee8ba3fd` |
| `swap.pybind.h` | `92d0859d8d2981d3267ef7e6f435dcd9f12b4b93` |

Shared width resolution preserves array order, singleton expansion and
empty views. The existing ordered swap dispatcher accepts quantum pairs
and same-basis classical pairs. Quantum lanes emit SWAP. Z and X
classical lanes each emit the three source XOR updates on their
underlying slots. Mixed columns dispatch lane by lane. A same-slot
classical lane clears that slot through the original three XOR updates.
Only a matching writable pair is accepted.

Fourteen locked streams cover the original SWAP cases, mixed X and Z
classical identities, both classical self-aliases and the empty case.
The mixed three-lane append test verifies one physical SWAP, six
classical inversions, the complete complex state and all sixteen input
classical assignments. There are no measurement branches or work wires.
The separate Qualtran swap resource declaration still models one
abstract Clifford per quantum pair.

The source appends valid leading lanes before throwing on a later
mixed-type mismatch. The observed witness retains `SWAP q0 q2` after
the second lane fails. Native compilation returns one complete
instruction tuple, so an invalid lane makes the whole append atomic.
Source finalization refuses duplicate quantum indices. Native rejects
them when constructing the instruction.

Run `python -m pytest -q tests/unit/test_kickmix_swap_binding_file.py`.
