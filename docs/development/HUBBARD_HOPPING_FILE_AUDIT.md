# Periodic Hubbard hopping file audit

This batch covers Q-0436 through Q-0438 in Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`. The three frozen file hashes matched. The original test file returned six passes and six skips.

| Inventory | Native evidence |
| --- | --- |
| Q-0436 | The original package initializer is documentary. The model and plaquette decomposition are recorded below. |
| Q-0437 | The three typed source signatures, immediate calls and resources are verified. Native bindings execute a four-mode plaquette and an independent color tile, with or without batched phasing. Source synthesis and decomposition gaps remain open. |
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
the source estimate. Symbolic HWP resource queries remain open in Q-0437;
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
