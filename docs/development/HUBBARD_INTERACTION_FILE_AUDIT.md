# Hubbard interaction file audit

Q-0440 and Q-0441 cover the on-site interaction declarations and original tests
from Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen sources matched their Git blobs. The unchanged source test file
returned six passes and six skips.

Both source variants declare one opaque through register named `system`
with shape `(2,)` and word width equal to the lattice side. The native
source signature matches the name, through direction, shape, ordered
indices and width for concrete and symbolic lengths. The concrete source
resource model derives its port width from this signature. The native
execution binds two row-major spin registers of length `length**2`.

The ordinary declaration repeats `Rz(angle * hubb_u)` once per site.
The HWP variant declares two `HammingWeightPhasing` calls of width
`length**2 // 2` and exponent `angle * hubb_u`. Source tests compare
these child parameters, synthesis precision, multiplicities, gate
categories and logical peaks. Symbolic signature support is independent
of concrete cost evaluation.

## Native interaction

The standalone operation reuses the on-site XOR basis and diagonal
emitter in the Suzuki step. It applies
`exp(i * timestep * hubb_u * sum(ZZ) / 4)` using `Rz(-timestep * hubb_u / 2)`
on each parity wire. The adjoint reverses the angle. HWP uses two equal
batches, preserves the scalar phase and reuses the same work and
feedback slots across batches. When the number of sites is odd, an
additional direct rotation covers the last parity wire, which is
omitted by the frozen two-batch cost formula. The one-site circuit
uses the ordinary direct rotation. A two-batch resource declaration
requires a positive batch width, so the one-site HWP binding is rejected.

For the two-by-two example, the ordinary native circuit emits eight CX
and four Rz with no work or measurements. The HWP variant emits eight
T-family instructions and two measurements, reusing three work wires.
The independent HWP source model gives a peak of five logical qubits.
Tests compare the ordinary full complex matrix in both directions
against the shifted-ZZ diagonal, and all four HWP measurement histories
against that diagonal on a coherent state. The full output vectors show
the work register restored to zero. A three-by-three sparse phase test
checks that the odd site's phase is present in both modes.

Q-0441 closes with the original checks and these native comparisons.
At that historical checkpoint Q-0440 retained the constructor, decomposition and synthesis boundaries.
Both original variants raise `DecomposeNotImplementedError`, and
the source model's angle parameter and width differ from the native
primitive angle and bound width. The exact native rotations use no
synthesis tolerance.

The complete register layouts for both classes now preserve the source's
default root `Interact..` and its indexed spin TextBoxes. The two distinct
class names truncate to the same label under the original drawing rule.
Tests cover one, two, three and eight sites per lattice side, ordered full
layout geometry, JSON round trips and rendered labels. These diagrams
describe source word ports rather than the native physical spin arrays.

The ordinary interaction now has an independent symbolic resource formula:
`L**2` copies of the classified `Rz(angle*hubb_u)`, with source peak
`Max(1, 2*L)`. For a positive integer symbolic L this peak simplifies to
`2*L`. Generic symbolic rotations, concrete zero, T, Clifford and general
angles match the unchanged source; concrete cases also match the existing
resource owner. This model emits no gates and retains the native physical
angle distinction above.

The source HWP symbolic gate query still fails on a missing `bit_length`,
and its peak query fails while splitting a symbolic dtype. The one-site
HWP source also fails its gate query by division by zero and its peak query
inside HammingWeightPhasing. These unchanged-source diagnostics are tested;
the ordinary formula is not substituted for HWP. Source decomposition,
precision, odd-site model and native phase differences were recorded against
Q-0440; the follow-up below preserves these source dispositions.

```python
from sympy import Symbol
from flagquantum.algorithms.primitives.hubbard_resource_model import (
    source_hubbard_interaction_formula,
)

length = Symbol("L", positive=True, integer=True)
angle = Symbol("theta", real=True)
model = source_hubbard_interaction_formula(length, angle, 2)
assert model.rotation.subs(length, 8) == 64
assert model.logical_qubits.subs(length, 8) == 16
```

```python
from flagquantum.algorithms.primitives.hubbard_lattice import HubbardInteraction
from flagquantum.dynamic import DynamicCircuit

op = HubbardInteraction.make_on(
    length=2,
    up=(0, 1, 2, 3),
    down=(4, 5, 6, 7),
    timestep=0.23,
    hubb_u=-1.7,
    hamming_weight=True,
    workspace=(8, 9, 10),
)
ir = DynamicCircuit(11).append(op).to_ir()
assert sum(gate.name == "measure" for gate in ir.instructions) == 2
```

## Complete constructor and dependency follow-up

The shared Compiler layout signature is now also the source of truth for
Algorithms' existing source signature helper: two opaque THRU words of width
L and shape (2,), in spin order. These ports own no work or feedback slots.
Physical interaction continues to bind two L**2-mode arrays, with its existing
caller-owned clean work only for the measured option. Formats binds the source
declarations; Algorithms owns modeling and the independently specified emitter.
No numerical execution or backend selection enters Compiler.

| Complete source definition | Native owner/disposition |
|---|---|
| Interaction and InteractionHWP fields and signatures | Both original constructors now resolve through shared port ownership, preserving length, angle, hubb_u and default/custom/symbolic eps metadata. |
| Interaction.build_call_graph | ComponentLibrary retains the Rz child with the exact angle*hubb_u expression, original precision and L**2 repetition count. |
| InteractionHWP.build_call_graph | ComponentLibrary retains two HWP children with floor(L**2/2) bits, the exact angle*hubb_u exponent and original eps; no order or missing primitive body is fabricated. |
| Both original factories and DocSpecs | Actual eight-side source examples, including angle0.52728 in the HWP factory, retain all constants and class/example associations. |
| Inherited wire symbols and both layouts | Existing complete Drawer geometry, indexed spin words, JSON and rendered labels reused unchanged. |

Native construction works without either reference SDK. When the original
protocol succeeds, its actual library resolves to the same typed bodyless
declaration as the native input graph. Such roots reject physical binding and
cannot silently select the independent shifted-ZZ numerical operator.
All immediate child identities, arguments, precision, signatures and counts
are compared with the unmodified original source in memory. The original
symbolic HWP protocol discovery failure remains explicit; source metadata is
still represented in the existing native input library.

Scalar Term bytes and decoded values match the actual original argument codec,
including its loss of symbol assumptions and Python-int decoding of SymPy
integer atoms. First-round normalization can reorder expressions; normalized
payloads are stable. These boundaries are not changed by the constructor work.
The one-site HWP declaration retains its zero-width child and empty signature,
while original resource/decomposition failures and independent physical
one-site admission remain distinct. No ordinary formula is substituted for a
failed HWP query. Original odd-site omission and source/native angle differences
remain recorded above.

The direct dependency uses the accepted HWP source constructor and its shared
count, weighted phase and exact measurement-assisted erasure owners. Source
negative profiles and unresolved HWP parent boundaries remain in that audit.
Existing full complex matrices in both directions, all four HWP measurement
histories with zero work, odd-site sparse phases and separate measured/model
resources are included in concentrated regression. Numerical emission and
resource arithmetic are unchanged by the signature ownership replacement.

Q-0440 closes the complete source file and its admitted direct dependency paths.
This is not a claim that the original model-only declaration supplies an ordered
primitive execution. Receipts use g2-hubbard-constructors-*-20261126.

Both source Suzuki factories also consume the new declarations without any
consumer change. Their ordinary/HWP and complete/stripped stored graphs export
the exact original operands, return wiring and four/five-call connections after
typed reconstruction. Physical binding continues to reject bodyless children.

| Two-by-two interaction | Measured native primitives/storage | Separate original estimate |
|---|---|---|
| Ordinary | 8 CX, 4 Rz; 8 system wires, zero work or measurements | 4 rotations, peak4 |
| HWP | 4 T + 4 Tdg, 8 phase, 44 CX, 2 CZ, 2 measurements; 8 system +3 work wires | 2 And, 26 Clifford, 4 rotations, 2 measurements, peak5 |

Feedback/reset and all other raw primitives remain in the resource receipt;
model categories and source word width do not describe that emitted stream.
Final concentrated regression passes 257 cases in each Cirq lane and local
integration passes 145. Both unmodified original files return 12 passes and
12 source decomposition-only skips. Strict five-file typing, formatting, lint,
both owner scopes, language/links, architecture, IR and public API checks pass.
Four source/test Git blobs, Qualtran0.7.0 and real Kickmix0.1 Simulator are verified.
