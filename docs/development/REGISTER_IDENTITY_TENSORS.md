# Direct-through register identity tensors (GRAPH59)

`simulation/tensor_network/register_identity.py` preserves direct input/output
connections for unused ports. It follows the per-bit identity placeholder loop
in the unchanged Qualtran v0.7.0 `_quimb._get_placeholder_tensors`, called by
`cbloq_to_quimb` for connections from LeftDangle directly to RightDangle.
The source is locked at bcb17bb25360ed68ce3bc084645db3e929799389.

## Interface and responsibility

`register_identity_nodes(signature, incoming, outgoing, like)` consumes existing
Core signatures and the GRAPH57 bit-label binding protocol. The caller passes
only those THRU ports known to be unused direct connections. For each declared
bit it returns an existing TensorNetworkNode containing a 2-by-2 identity,
with output label before input label. It does not discover graph connectivity,
allocate labels, reset data, or insert itself on operated ports. A caller that
inserts placeholders twice has supplied an invalid graph.

The existing binder owns row-major word order, MSB-first bit order and label
validation. Native additionally rejects directional ports and external aliases:
input/output labels must be distinct, including across different bits. The
source placeholder's synthetic connection objects create distinct external
ends; native explicit integer labels preserve that distinction without copying
foreign Soquet or Connection classes. Missing live bindings raise KeyError;
shape/width/label and alias errors raise ValueError before tensor allocation.

Each identity takes its dtype/device from the caller's `like` tensor. Storage
is linear in bit count; no `2**n` word identity is built. Zero-width and
empty-array ports return no nodes, matching the source bit loop. A scalar-only
network still follows the existing contraction interface's requirement for an
explicit scalar node. These bit placeholders do not imply an automatic empty
network identity or change scalar contraction behavior.

Simulation owns these numerical nodes. Compiler remains responsible for graph
transformations and physical workspace. There is no new graph type, backend,
provider registry, compiler primitive or protected API. The helper emits no
gates and has no logical-resource estimate.

## Source and execution evidence

Sixteen width/shape cases cover widths zero through three and scalar, vector,
matrix-shaped and zero-length arrays. The actual original `cbloq_to_quimb`
produces the same number of per-bit identity tensors. For nonempty cases,
`quimb_to_dense` and native contraction produce the same complete operator.
No-bit cases separately assert that both paths produce no placeholder nodes.

A controlled X under all four two-control activation patterns is paired with
an unused two-bit integer register. The active node uses GRAPH56's dense
controlled tensor; exactly two identity nodes preserve the unused word. The
result matches the unchanged source's complete `cbloq_to_quimb` network and
`quimb_to_dense` output. Replacing the active node with identity preserves the
unused-register consumer and yields the full identity operator.

The existing `circuit.append(op)` measured predicate emitter independently
executes the same logical operation. Original source gate names, operands,
parameters and feedback match. Across four activation patterns, all eight
forward measurement branches and sixteen forward/inverse history pairs retain
complete complex amplitudes, including correlations with the unused word, and
return the clean work wire to zero. Sixteen forward plus sixteen roundtrip
DynamicCircuit trajectories pass. The unused word is preserved, not projected
onto zero or reconstructed from probabilities.

Per circuit the emitted stream has four T/Tdag and one measurement/reset,
using one work wire. Independent source QEC models report four T-equivalents
and one measurement. Source active logical peak four excludes the two unused
bits; the complete composite peak and caller width include them and are six.
Full primitive counts and seven source model categories remain separate from
the two identity tensor nodes. Numerical placeholders do not count as logical
quantum gates.

Four real/complex PyTorch dtypes preserve device/dtype and the analytical
squared-amplitude gradient through an attached ket. No NumPy conversion or
detachment occurs in the production helper.

Run the focused tests in the locked reference environment:

```bash
python -m pytest tests/unit/test_register_identity_tensors.py -q
```

Automatic whole-Program connectivity discovery, superoperator placeholders,
foreign-friendly index names, and complete `_quimb` parent-file acceptance
remain open. This finite slice does not claim a complete graph-to-tensor runner.

## Final local verification

Implementation commit: `88b4503c`. The original eight register identity-gate
tests remain byte-for-byte unchanged; the thirty tensor tests live separately
in `tests/unit/test_register_identity_tensors.py`.

- Related regression: 857 passed in 9.20 seconds.
- Focused coverage: 162 passed in 6.14 seconds; five selected owners have
  135 statements and 54 branches at 100%. The new owner contributes
  16 statements and six branches, all covered.
- Integration: 64 passed in 3.33 seconds.
- Strict typing, Black and Ruff passed for ten selected files. Simulation scope,
  architecture, Program contract and protected API checks passed.
- Ten original source blobs match the locked inventory; two unchanged upstream
  quimb tests passed in 0.87 seconds.
- Locked environment confirmed Qualtran 0.7.0 and Kickmix 0.1 with Simulator.
  The existing binary-field arithmetic example passed, including sixteen
  measured product-erasure trajectories and clean inverse workspace.

The final default smoke/unit run reports 21,507 passed, 300 skipped and seven
failed in 116.39 seconds. All seven failure IDs exactly match the independent
existing baseline; no new failure ID appeared. The default suite is not green.
The earlier run before restoring the original identity-gate test filename is
superseded and is not final acceptance evidence. Logs and the exact baseline
comparison are stored under `ftqc-reference-env/g2-register-identity-*` in the
local reference workspace.
