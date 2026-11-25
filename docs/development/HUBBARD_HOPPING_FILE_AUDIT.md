# Periodic Hubbard hopping file audit

This batch covers Q-0436 through Q-0438 in Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`. The three frozen file hashes matched. The original test file returned six passes and six skips.

| Inventory | Native evidence |
| --- | --- |
| Q-0436 | The original package initializer is documentary. The model and plaquette decomposition are recorded below. |
| Q-0437 | Complete source constructors, signatures, call metadata, examples, drawing and models are mapped by the follow-up below. Native execution remains separately specified; unavailable source decomposition and synthesis are explicit dispositions. |
| Q-0438 | The source examples, original count tests and native matrix and measurement-branch comparisons cover the original test file. |

## Hamiltonian and ports

The periodic square-lattice Hubbard Hamiltonian is $H=H_h+H_I$.
For each spin, $H_h$ is minus the hopping coupling times the sum of
nearest-neighbor fermionic hopping operators. The shifted interaction is
$H_I=(u/4)\sum_p Z_{p\uparrow}Z_{p\downarrow}$, with the occupation formula
$Z_{p\sigma}=2n_{p\sigma}-1$ used in the source documentation. The two
periodic color coverings partition the nearest-neighbor edges and permit
independent evolution within each color. For a two-by-two periodic lattice
the two colors retain separate contributions from parallel bonds.

The declared plaquette has a through array of four QBits. Both source tile
variants expose one shaped through register containing two opaque words of
width $L$. The executable lattice uses two row-major spin arrays of
$L^2$ occupation wires each. The source metadata keeps its declared
width for concrete and symbolic $L$, including the ordered spin indices.
The native binding validates the full arrays and reserves clean work only
for the measured phasing option.

## Source call model and native execution

The source plaquette declares four `TwoBitFFFT(0, 1)` calls and two
`Rz(kappa)` calls. An ordinary tile repeats the plaquette $L^2/2$ times
with `kappa = tau * angle`. The HWP model instead declares $2L^2$
Fourier calls followed by two equal-exponent phasing batches on $L^2/2$
modes. Source tests pin the angles, Fourier indices, precision forwarding
and multiplicities. The concrete resource model classifies the declared
rotations and records gate totals and peak qubits.

The native standalone tile uses the same fermionic routing and the same
diagonal phase layer as the Suzuki step. It binds `exp(i * timestep * hubb_t * K)`
for the selected color adjacency operator `K`. The diagonal implementation
uses `Rz(-2 * timestep * hubb_t)` with the appropriate basis change. It
preserves absolute phase under HWP by including the scalar offset of the
two batched phase passes. An adjoint flips the time sign. HWP reuses the
existing measurement slots, with clean work restored after both passes.

On a two-by-two lattice the ordinary native operator emits 16 T-family
instructions and uses eight system wires; its source model peak is four
logical qubits. The HWP version emits 24 T-family instructions and two
measurements and binds three additional work wires. The HWP test compares
all four measurement histories to the independent Jordan–Wigner matrix
and checks complex amplitudes including work-wire zero outcomes.
Separate four-by-four sparse execution checks the two distinct periodic
color cycles with both spin sectors present.

The pinned plaquette and tile graphs do not provide an ordered full
primitive execution. Their `eps` is a source synthesis parameter; the
native exact rotations have no synthesis tolerance in their binding.
The native routed sequence and its measured costs are kept separate from
the source estimate. Symbolic HWP resource queries remain unsupported upstream;
symbolic lattice sizes do not denote an executable finite circuit.

The complete source register layouts for all three classes now use the
shared Hubbard drawer. The plaquette inherits the truncated `HoppingP..`
default root. Both ordinary and HWP tiles use `H_h^p` or `H_h^g` according
to color. Although HWP defines a short name with an `(HWP)` suffix, its
inherited wire_symbol does not use that suffix. Tests compare the source
geometry, ordered indexed TextBoxes and root attributes for lengths 2, 4
and 8, both colors, JSON round trips and rendered labels. Existing
plaquette and tile matrix, primitive, feedback and work-cleanup evidence
is reused. This presentation binding changes no emitted gates.

The ordinary tile now has an independent symbolic resource formula. It
composes the shared two-bit Fourier costs with the classified plaquette
rotation and multiplies by `floor(L**2/2)`. For a generic symbolic angle
the locked QEC estimate is `8*floor(L**2/2)` T gates,
`52*floor(L**2/2)` Cliffords, and `2*floor(L**2/2)` rotations, with logical
peak `Max(4, 2*L)`. Concrete zero, T, Clifford and generic angles are checked
against both the unmodified source and the existing concrete model.
The formula estimates gates only; it emits no symbolic circuit.

The historical open source constructor gap is closed by the follow-up below;
the original HWP failures remain preserved rather than replaced by the ordinary
formula.

The unchanged HWP symbolic source fails its gate query with an
AttributeError on the phasing width's missing `bit_length`, and its width
query with a RuntimeError from splitting a symbolic dtype. Tests retain
both diagnostics. No ordinary formula is substituted for HWP. Its existing
concrete costs, numerical execution and branch evidence remain valid.

```python
from sympy import Symbol
from flagquantum.algorithms.primitives.hubbard_resource_model import (
    source_hubbard_hopping_tile_formula,
)

length = Symbol("L", positive=True, integer=True)
angle = Symbol("theta", real=True)
model = source_hubbard_hopping_tile_formula(length, angle)
assert model.t.subs(length, 8) == 256
assert model.rotation.subs(length, 8) == 64
```


## Run one color

```python
from flagquantum.algorithms.primitives.hubbard_lattice import HubbardHoppingTile
from flagquantum.dynamic import DynamicCircuit

tile = HubbardHoppingTile.make_on(
    length=2,
    up=(0, 1, 2, 3),
    down=(4, 5, 6, 7),
    timestep=0.21,
    hamming_weight=True,
    workspace=(8, 9, 10),
)
circuit = DynamicCircuit(11).append(tile)
assert sum(gate.name == "measure" for gate in circuit.to_ir().instructions) == 2
```

## Complete constructor and dependency follow-up

The path classification is single_device_fast_path. Compiler layout owns the
shared x/y Fourier pair, four-mode plaquette array and two opaque L-bit spin
words in registers/layout/fermionic_signatures.py. Existing Algorithms helpers
delegate to those same signatures; resource classifications and numerical
emission do not move into Compiler. Formats owns source constructor binding
and data-bearing ComponentLibrary declarations in legacy_hubbard_factories.py.
There are no work bits or measurements on these source through ports. The
separately selected physical tile retains its two row-major L**2 spin arrays
and caller-owned measured-phasing workspace.

| Complete source definition | Native owner/disposition |
|---|---|
| HoppingPlaquette fields, signature and build_call_graph | Formats retains original kappa/eps values and shared four-mode ports; direct metadata preserves four TwoBitFFFT(0,1,eps) and two Rz(kappa,eps) calls. |
| HoppingTile fields and __attrs_post_init__ | Formats preserves length/angle/tau/eps/pink defaults, and the original Python-int-only odd-length rejection. Scalars are validated without changing integer constructor arguments into floats. |
| HoppingTile signature and build_call_graph | Shared two-spin-word signature and one plaquette child with kappa=tau*angle, original precision and floor(L**2/2) multiplicity. |
| HoppingTile.wire_symbol | Existing complete Drawer geometry, indexed spin words, both colors, JSON and rendering evidence reused unchanged. |
| HoppingTileHWP.short_name | The default input-library label preserves H_h^p(HWP) or H_h^g(HWP); an explicit caller label overrides it. The inherited wire symbol still has no HWP suffix, exactly as upstream. |
| HoppingTileHWP.build_call_graph | Metadata retains floor(4*L**2/2) Fourier calls and two HWP children of width floor(2*L**2/4), exponent tau*angle and unchanged eps. No execution order is invented. |
| All three original examples and DocSpecs | Actual source objects, class associations and original constants are checked. The original _plaquette uses kappa=8 and eps=0.15; these are not corrected to the apparent intent of its local variable names. |

All three classes resolve from actual original serialized constructor libraries
when the source protocol succeeds. Their native input libraries also compare
every direct child's qualified identity, fields, precision, port order and
multiplicity with the unmodified in-memory source. Bodyless typed roots reject
physical binding; they never become identity operations. Independent plain/HWP
tile execution still uses circuit.append(op) through its existing owner.

Symbolic HWP protocol discovery recursively queries the child and fails on its
missing bit_length. The in-memory source call metadata remains available and
the existing native ComponentLibrary can preserve it. Legacy scalar Term
serialization discards symbol assumptions and reads SymPy integer atoms as
Python ints in both codecs; direct argument bytes and decoded values are
compared against the actual original codec. The first normalization can change
expression ordering or the scalar encoding arm, and subsequent normalized
bytes are stable. In particular SymPy Integer(3) bypasses the original
constructor parity guard, while its decoded Python 3 is rejected by both the
original constructor and native resolution. This is an explicit protocol
boundary, not a claim of lossless integer-class or assumption preservation.

Direct dependencies reuse the accepted forward TwoBitFFFT constructor/body and
rotation owner, and the existing HWP count/phase/exact-erasure closure. Their
independent outstanding parent defects remain listed in their own audits;
the hopping declaration does not invent missing source primitive bodies.
Existing complete complex plaquette/tile matrices, both phase directions,
all two-bit measurement histories with zero work, larger periodic color routes
and separate primitive/model counts are included in concentrated regression.
The numerical emitter and resource arithmetic remain unchanged by this batch.

Q-0437 closes this complete file and its admitted direct dependency paths.
The source-only word width, model-only order, synthesis metadata, HWP symbolic
failures and independent numerical phase convention remain explicit.
Receipts use g2-hubbard-constructors-*-20261126.

The unchanged Suzuki consumer resolves both ordinary/HWP variants, with or
without the stripped layer, through the existing generic resolver. Exporting
the reconstructed typed Program reproduces the original stored connections,
instance operands, return wiring and four/five-call order exactly. The leaf
roots remain bodyless and refuse physical binding. Existing consumers and the
independent numerical emitters require no modifications.

| Fixture | Measured native primitives/storage | Separate original estimate |
|---|---|---|
| Four-mode plaquette | 4 T + 4 Tdg, 6 Rz, 16 CX; 4 system wires, no work or measurement | 8 T, 52 Clifford, 2 rotations, peak4 |
| Two-by-two ordinary tile | 8 T + 8 Tdg, 12 Rz, 32 CX; 8 system wires | 16 T, 104 Clifford, 4 rotations, peak4 |
| Two-by-two HWP tile | 12 T + 12 Tdg, 8 Rz, 8 phase, 68 CX, 2 measurements; 8 system +3 work wires | 16 T, 2 And, 130 Clifford, 4 rotations, 2 measurements, peak5 |

The raw primitive receipt also records swaps, CZ, feedback and reset. The model
does not predict that independent routed stream. Final concentrated regression
passes 257 cases in each Cirq lane; local integration passes 145. The two
unmodified source test files return 12 passes and 12 original decomposition-only
skips. Strict typing covers all five changed Python files; format, lint, both
owner scopes, language/links, architecture, Program IR and public API checks pass.
All four source/test Git blob IDs match the locked commit, with Qualtran0.7.0
and real Kickmix0.1 Simulator verified. No numerical or resource arithmetic was
changed by moving signature ownership.
